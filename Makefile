MAIN='main.cpp' #Put your path to a file containing `main` here
OUTPUT_EXECUTABLE_NAME='a'
EIGEN_DIRECTORY_PATH='eigen3_4_0/' #Put your path to the Eigen 3 folder here

c:
	g++ ${MAIN}  -o ${OUTPUT_EXECUTABLE_NAME}  -std=c++14  -I ${EIGEN_DIRECTORY_PATH}