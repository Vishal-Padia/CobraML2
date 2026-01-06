// kernel 1 (QK^T):
// Grid: (ceil(N/TILE), ceil(N/TILE), B*H)
// Block: (TILE, TILE)

// kernel 2 (Softmax):
// Grid: (B*H*N) --> one block per row
// Block: (min(N, 1024), ) --> or use multiple warps

// kernel 3 (PV):
// Grid: (ceil(N/TILE), ceil(d/TILE), B*H)
// Block: (TILE, TILE)

// Output = softmax(Q @ K^T / sqrt(d)) @ V
// Q, K, V: [B, H, N, d] - batch, heads, sequence length, head dimension
// attention matrix S = Q @ K^T has shape [B, H, N, N]
// the output O has shape [B, H, N, d]

#pragma once
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace cobraml::kernels {

using namespace cute;

// helper functions
template <int TILE_ROW, typename DType>
CUTE_HOST_DEVICE auto make_gemm_tiled_copy() {
  return make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, DType>{},
      Layout<Shape<Int<TILE_ROW>, Int<16>>>{});
}

template <int TILE_M, int TILE_N, typename DType>
CUTE_HOST_DEVICE auto make_gemm_tiled_mma() {
  return make_tiled_mma(UniversalFMA<DType, DType, DType>{},
                        Layout<Shape<Int<TILE_M>, Int<TILE_N>, _1>>{});
}

namespace mha_cute {
template <int TILE_N, int HEAD_DIM, int NUM_HEADS, typename DType, typename TiledCopyQ,
          typename TiledCopyK, typename TiledMMA>
__global__ void qk_kernel(const DType *__restrict__ Q, // [B, H, N, d]
                          const DType *__restrict__ K, // [B, H, N, d]
                          DType *__restrict__ S,       // [B, H, N, N]
                          int B, int N, TiledCopyQ tiled_copy_q,
                          TiledCopyK tiled_copy_k, TiledMMA tiled_mma) {
  // Create global memory tensor
  auto Q_tensor = make_tensor(
      make_gmem_ptr(Q),
      make_layout(make_shape(B, Int<NUM_HEADS>{}, N, Int<HEAD_DIM>{}), LayoutRight{}));
  auto K_tensor = make_tensor(
      make_gmem_ptr(K),
      make_layout(make_shape(B, Int<NUM_HEADS>{}, N, Int<HEAD_DIM>{}), LayoutRight{}));
  auto S_tensor = make_tensor(
      make_gmem_ptr(S), make_layout(make_shape(B, Int<NUM_HEADS>{}, N, N), LayoutRight{}));

  // Decode batch and head indices from blockIdx.z
  int bh = blockIdx.z;
  int b_idx = bh / NUM_HEADS;
  int h_idx = bh % NUM_HEADS;

  // Slice tensors to current batch and head
  auto Q_bh = Q_tensor(b_idx, h_idx, _, _); // [N, HEAD_DIM]
  auto K_bh = K_tensor(b_idx, h_idx, _, _); // [N, HEAD_DIM]
  auto S_bh = S_tensor(b_idx, h_idx, _, _); // [N, N]

  // each cta handles one of S tiles (tile_row is fixed),
  int tile_row = blockIdx.y;

  // extract gQ for this row
  auto gQ = local_tile(Q_bh, make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}),
                       make_coord(tile_row, 0));
  
  // shared memory layouts
  auto sQ_layout = make_layout(make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}), LayoutRight{});
  auto sK_layout = make_layout(make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}), LayoutRight{});

  __shared__ DType smem_q[cosize_v<decltype(sQ_layout)>];
  __shared__ DType smem_k[cosize_v<decltype(sK_layout)>];

  Tensor sQ = make_tensor(make_smem_ptr(smem_q), sQ_layout);
  Tensor sK = make_tensor(make_smem_ptr(smem_k), sK_layout);

  auto thr_copy_q = tiled_copy_q.get_slice(threadIdx.x);
  auto thr_copy_k = tiled_copy_k.get_slice(threadIdx.x);

  // partition Q for copy
  Tensor tQgQ = thr_copy_q.partition_S(gQ);
  Tensor tQsQ = thr_copy_q.partition_D(sQ);

  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCsQ = thr_mma.partition_A(sQ);
  Tensor tCsK = thr_mma.partition_B(sK);

  // load Q once outside loop
  copy(tiled_copy_q, tQgQ, tQsQ);
  __syncthreads();
  
  // number of k tiles to iterate over
  int num_k_tiles = N / TILE_N;

  DType scale = DType(1.0) / sqrt(DType(HEAD_DIM));

  for (int tile_col = 0; tile_col < num_k_tiles; ++tile_col) {
    // Get K tile for this iteration
    auto gK = local_tile(K_bh, make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}),
                         make_coord(tile_col, 0));

    // Get S tile for output
    auto gS = local_tile(S_bh, make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
                         make_coord(tile_row, tile_col));

    // Partition K for copy
    Tensor tKgK = thr_copy_k.partition_S(gK);
    Tensor tKsK = thr_copy_k.partition_D(sK);

    // Load K tile (streamed each iteration)
    copy(tiled_copy_k, tKgK, tKsK);
    __syncthreads();

    // Partition S for output
    Tensor tCgS = thr_mma.partition_C(gS);
    Tensor tCrS = thr_mma.make_fragment_C(tCgS);
    clear(tCrS);

    // Compute GEMM: S_tile = Q_tile @ K_tile^T
    gemm(tiled_mma, tCsQ, tCsK, tCrS);

    // Apply scaling and write
    CUTE_UNROLL
    for (int i = 0; i < size(tCrS); ++i) {
      tCrS(i) *= scale;
    }

    copy(tCrS, tCgS);
    __syncthreads();
  }
}

template <int BLOCK_SIZE, typename DType>
__global__ void softmax_kernel(DType *__restrict__ S, // [B, H, N, N]
                               int B, int H, int N) {
  // each block handles one row
  int row_idx = blockIdx.x;
  int total_rows = B * H * N;
  if (row_idx >= total_rows)
    return;

  // decode (b, h, i) from flattened row index
  int b = row_idx / (H * N);
  int rem = row_idx % (H * N);
  int h = rem / N;
  int i = rem % N;

  // create tensor views
  auto S_tensor = make_tensor(
      make_gmem_ptr(S), make_layout(make_shape(B, H, N, N), LayoutRight{}));

  // get this row
  auto S_row = S_tensor(b, h, i, _);

  // shared memory for parallel reduction
  __shared__ DType smax[BLOCK_SIZE];
  __shared__ DType ssum[BLOCK_SIZE];

  int tid = threadIdx.x;

  // find row maximum
  DType thread_max = -INFINITY;
  for (int j = tid; j < N; j += BLOCK_SIZE) {
    thread_max = fmaxf(thread_max, S_row(j));
  }
  smax[tid] = thread_max;
  __syncthreads();

  // parallel reduction to find global max
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
    }
    __syncthreads();
  }
  DType row_max = smax[0];

  // compute exp(x - max) and sum
  DType thread_sum = DType(0);
  for (int j = tid; j < N; j += BLOCK_SIZE) {
    thread_sum += expf(S_row(j) - row_max);
  }
  ssum[tid] = thread_sum;
  __syncthreads();

  // parallel reduction to find sum
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      ssum[tid] += ssum[tid + stride];
    }
    __syncthreads();
  }
  DType row_sum = ssum[0];

  // write normalized softmax values
  for (int j = tid; j < N; j += BLOCK_SIZE) {
    S_row(j) = expf(S_row(j) - row_max) / row_sum;
  }
}

template <int TILE_N, int HEAD_DIM, int NUM_HEADS, typename DType, typename TiledCopyP,
          typename TiledCopyV, typename TiledMMA>
__global__ void pv_kernel(const DType *__restrict__ P, // [B, H, N, N]
                          const DType *__restrict__ V, // [B, H, N, d]
                          DType *__restrict__ O,       // [B, H, N, d]
                          int B, int N, TiledCopyP tiled_copy_p,
                          TiledCopyV tiled_copy_v, TiledMMA tiled_mma) {
  // create global memory tensor views
  auto P_tensor = make_tensor(
      make_gmem_ptr(P), make_layout(make_shape(B, Int<NUM_HEADS>{}, N, N), LayoutRight{}));
  auto V_tensor = make_tensor(
      make_gmem_ptr(V), make_layout(make_shape(B, Int<NUM_HEADS>{}, N, Int<HEAD_DIM>{}), LayoutRight{}));
  auto O_tensor = make_tensor(
      make_gmem_ptr(O), make_layout(make_shape(B, Int<NUM_HEADS>{}, N, Int<HEAD_DIM>{}), LayoutRight{}));

  // decode batch and head indices
  int bh = blockIdx.z;
  int b_idx = bh / NUM_HEADS;
  int h_idx = bh % NUM_HEADS;

  // slice to current batch and head
  auto P_bh = P_tensor(b_idx, h_idx, _, _); // [N, N]
  auto V_bh = V_tensor(b_idx, h_idx, _, _); // [N, d]
  auto O_bh = O_tensor(b_idx, h_idx, _, _); // [N, d]

  // cta tile coordinates
  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  // shared memory layouts
  auto sP_layout =
      make_layout(make_shape(Int<TILE_N>{}, Int<TILE_N>{}), LayoutRight{});
  auto sV_layout =
      make_layout(make_shape(Int<TILE_N>{}, Int<TILE_N>{}), LayoutRight{});

  __shared__ DType smem_p[cosize_v<decltype(sP_layout)>];
  __shared__ DType smem_v[cosize_v<decltype(sV_layout)>];

  Tensor sP = make_tensor(make_smem_ptr(smem_p), sP_layout);
  Tensor sV = make_tensor(make_smem_ptr(smem_v), sV_layout);

  // thread partitioning for copy
  auto thr_copy_p = tiled_copy_p.get_slice(threadIdx.x);
  auto thr_copy_v = tiled_copy_v.get_slice(threadIdx.x);

  // thread partitioning for mma
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  // number of tiles along the reduction dimension
  int num_j_tiles = N / TILE_N;

  // get output tile and allocate accumulator
  auto gO = local_tile(O_bh, make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
                       make_coord(tile_row, tile_col));

  Tensor tCgO = thr_mma.partition_C(gO);
  Tensor tCrO = thr_mma.make_fragment_C(tCgO);
  clear(tCrO);

  // accumulation loop over N dimension
  for (int j_tile = 0; j_tile < num_j_tiles; ++j_tile) {
    // get P tile: [TILE_N, TILE_N]
    auto gP = local_tile(P_bh, make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
                         make_coord(tile_row, j_tile));

    // get V tile: [TILE_N, TILE_N]
    auto gV = local_tile(V_bh, make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
                         make_coord(j_tile, tile_col));

    // Partition for copy
    Tensor tPgP = thr_copy_p.partition_S(gP);
    Tensor tVgV = thr_copy_v.partition_S(gV);

    Tensor tPsP = thr_copy_p.partition_D(sP);
    Tensor tVsV = thr_copy_v.partition_D(sV);

    // Load P and V tiles to shared memory
    copy(tiled_copy_p, tPgP, tPsP);
    copy(tiled_copy_v, tVgV, tVsV);

    __syncthreads();

    // Partition shared memory for MMA
    Tensor tCsP = thr_mma.partition_A(sP);
    Tensor tCsV = thr_mma.partition_B(sV);

    // Compute P @ V and accumulate
    gemm(tiled_mma, tCsP, tCsV, tCrO);

    __syncthreads();
  }

  copy(tCrO, tCgO);
}
} // namespace mha_cute

template <int TILE_N = 16, int HEAD_DIM = 64, int NUM_HEADS = 8, 
          int SOFTMAX_BLOCK = 256, typename DType = float>
void mha_forward(DType *Q, DType *K, DType *V, DType *O, int B, int N) {
  // allocate intermediate buffer (S will hold scores, then probabilities after softmax)
  DType *S;
  cudaMalloc(&S, B * NUM_HEADS * N * N * sizeof(DType));

  {
    auto tiled_copy_q = make_gemm_tiled_copy<TILE_N, DType>();
    auto tiled_copy_k = make_gemm_tiled_copy<TILE_N, DType>();
    auto tiled_mma = make_gemm_tiled_mma<TILE_N, TILE_N, DType>();

    constexpr int num_threads = TILE_N * TILE_N;

    dim3 grid(1, (N + TILE_N - 1) / TILE_N, B * NUM_HEADS);
    dim3 block(num_threads);

    mha_cute::qk_kernel<TILE_N, HEAD_DIM, NUM_HEADS, DType><<<grid, block>>>(
        Q, K, S, B, N, tiled_copy_q, tiled_copy_k, tiled_mma);
  }

  {
    int total_rows = B * NUM_HEADS * N;
    mha_cute::softmax_kernel<SOFTMAX_BLOCK, DType>
        <<<total_rows, SOFTMAX_BLOCK>>>(S, B, NUM_HEADS, N);
  }

  {
    auto tiled_copy_p = make_gemm_tiled_copy<TILE_N, DType>();
    auto tiled_copy_v = make_gemm_tiled_copy<TILE_N, DType>();
    auto tiled_mma = make_gemm_tiled_mma<TILE_N, TILE_N, DType>();

    constexpr int num_threads = TILE_N * TILE_N;

    dim3 grid((HEAD_DIM + TILE_N - 1) / TILE_N, (N + TILE_N - 1) / TILE_N, B * NUM_HEADS);
    dim3 block(num_threads);

    // Use S directly - it now contains softmax probabilities
    mha_cute::pv_kernel<TILE_N, HEAD_DIM, NUM_HEADS, DType><<<grid, block>>>(
        S, V, O, B, N, tiled_copy_p, tiled_copy_v, tiled_mma);
  }

  cudaDeviceSynchronize();
  cudaFree(S);
}
} // namespace cobraml::kernels