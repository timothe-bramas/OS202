#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include   <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
int i,j,k;
//#pragma omp parallel for private(i,j,k) shared(A,B,C) num_threads(3)

  for (k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
    for (j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)   
      for (i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
        C(i, j) += A(i, k) * B(k, j);
}

const int szBlock = 16;
}  // namespace


Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  int ii,jj,kk;
  int n=A.nbRows/szBlock;

  for (kk = 0; kk < n; kk+=szBlock)
    for (jj = 0; jj < n; jj+=szBlock) 
      for (ii = 0; ii < n; ii+=szBlock) 
        prodSubBlocks(ii,jj,kk,szBlock,A,B,C);
    //prodSubBlocks(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
  return C;
}
