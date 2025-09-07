MAIN='main.cpp' #Put your path to a file containing `main` here
OUTPUT_EXECUTABLE='a'
EIGEN_PATH='eigen3_4_0/' #Put your path to the Eigen 3 folder here

c:
	g++ ${MAIN}  -o ${OUTPUT_EXECUTABLE}  -std=c++14  -I ${EIGEN_PATH}