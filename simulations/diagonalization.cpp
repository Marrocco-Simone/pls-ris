#include <iostream>
#include <Eigen/Dense>

// using Eigen::MatrixXd;

int main()
{
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 4;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;
}