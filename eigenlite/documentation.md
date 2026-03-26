# Eigen Lite Documentation

[Back to central documentation](../documentation/documentation.md)

Documentation for the minimal version of Eigen used in this package.

<details>
  <summary>Details</summary>
  
This file is located in `eigenlite/documentation.md`.

</details>
<br>

Eigen Lite implements only the functionality needed for CNet.

All Eigen Lite functionality is under the `Eigen` namespace, identical to that of the full Eigen distribution.  
Eigen Lite defines `USING_EIGENLITE`, used for conditional compilation in case Eigen-specific functionality is used. Use conditional compilation checking for `USING_EIGENLITE` to choose between standard Eigen and Eigen Lite.

## Table of Contents
- [Matrix](#matrix)
    - [Constructors](#constructors)
    - [Getters](#getters)
    - [Setters](#setters)
    - [Arithmetic Operator Overloads](#arithmetic-operator-overloads)
    - [Other Operator Overloads](#other-operator-overloads)

---
---
---

## Matrix

Partial implementation of Eigen's matrix class. All features are under the `Eigen` namespace.

This class lacks some methods available in the full Eigen distribution, especially `dot` and `array`.

A Matrix has 2 static arguments:
- `T`- the Matrix's datatype
- `static_type`- of type `MatrixStaticType`. `MatrixStaticType` is an enum with 2 possible values: `MATRIX` (statically typed matrix, enum value 0) and `COLUMN_VECTOR` (statically typed column vector, enum value 1).

Example:
```
using namespace Eigen;

//Matrix of type double and static type MATRIX
Matrix<double, MATRIX> m;

//Matrix of type int and static type COLUMN_VECTOR
Matrix<int, COLUMN_VECTOR> v;
```

Some methods may be called only for static type `COLUMN_VECTOR`.

A Matrix supports comma initialization, where a comma-separated list of values, in row-major order, sets a Matrix's value.  
Example:
```
using namespace Eigen;

//Initializes matrix to [[1, 2], [3, 4]]
Matrix<double, MATRIX> m(2, 2);
m << 1, 2, 3, 4;
```

A Matrix's elements are accessed using the parentheses operator. Elements have 0-based indexes.  
Example:
```
Given a Matrix `m`, initialized to
[[1  2]
 [3  4]]

std::cout << m(0, 0); //prints 1
std::cout << m(1, 1); //prints 4
```

Matrices of static type COLUMN_VECTOR can be accessed using a single-argument parentheses operator, with 0-based indices.  
Example:
```
Given a Matrix<T, COLUMN_VECTOR> `v`, initialized to
[[1  ]
 [10 ]
 [100]]

std::cout << v(0); //prints 1
std::cout << v(2); //prints 100
```

The Matrix class has 2 typedefs: `MatrixXd` for `Matrix<double, MATRIX>`, and `VectorXd` for `Matrix<double, COLUMN_VECTOR>`.

<br>
<br>

Example usage
```
using namespace Eigen;

//Create 2x3 matrix, of type double, initialized to [[1, 2, 3], [4, 5, 6]]
Matrix<double, MATRIX> m1(2, 3);
m1 << 1, 2, 3, 4, 5, 6;

//Create 3x2 matrix, of type double, initialized to [[1, 2], [3, 4], [5, 6]]
MatrixXd m2(3, 2);
m2 << 1, 2, 3, 4, 5, 6;

//Create 1x3 statically-typed column vector, initialized to [[1], [2], [3]]
VectorXd v(3); //equivalently: Matrix<double, COLUMN_VECTOR>(3);
v << 1, 2, 3;


//Get elements from matrix m1
std::cout << m1(0, 0) << std::endl; //1
std::cout << m1(1, 2) << std::endl; //6

//Get elements from the vector v
std::cout << v(0) << std::endl; //1
std::cout << v(2) << std::endl; //3


//Calculate matrix-vector product
std::cout << (m1 * v) << std::endl; //[[14], [32]]

//Calculate matrix-matrix product
std::cout << (m1 * m2) << std::endl; //[[22, 28], [49, 64]]

//Calculate matrix-scalar product
std::cout << (m1 * 3) << std::endl; //[[3, 6, 9], [12, 15, 18]]
```

---
---

### Constructors

#### Default Constructor

*Signature:* `Matrix()`

Creates an empty, uninitialized matrix.

Initializes the matrix with no contents and sets the number of rows and columns to zero. It does not allocate memory for matrix elements.

---

#### Rows and columns

*Signature:* `Matrix(int32_t n_rows, int32_t n_cols)`

Creates a matrix with `n_rows` rows and `n_cols` columns. The contents are uninitialized.

For matrices only. Works for statically-typed vectors only if `n_cols` is 1.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the matrix. Must be positive.
* `n_cols` (`int32_t`): Number of columns in the matrix. Must be positive.

---

#### Rows (Column Vector)

*Signature:* `Matrix<T, COLUMN_VECTOR>(int32_t n_rows)`

Creates a column vector with `n_rows` rows.

For statically typed column vectors only.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the new vector. Must be positive.

---
---

### Getters

#### at (Matrix)

*Signature:* `const T& at(int32_t row_index, int32_t col_index) const`

Returns a constant reference to the matrix element at position `(row_index, col_index)`.

Value indexing is 0-based. The top left value is at position (0, 0).

**Returns**

* `const T&`: Constant reference to the matrix element at position `(row_index, col_index)`.

**Parameters**

* `row_index` (`int32_t`): The row index. Must satisfy `0 <= row_index < rows()`.
* `col_index` (`int32_t`): The column index. Must satisfy `0 <= col_index < cols()`.

---


#### at (Column Vector)

*Signature:* `const T& at(int32_t row_index) const`

Returns a constant reference to the element at position `row_index`.

Value indexing is 0-based. The top value is at position 0.

For statically-typed column vectors only.

**Returns**

* `const T&`: Constant reference to the matrix element at position `row_index` in a column vector.

**Parameters**

* `row_index` (`int32_t`): The row index. Must satisfy `0 <= row_index < rows()`.

---

#### cols

*Signature:* `int32_t cols() const`

Returns the number of columns in the matrix.

**Returns**

* `int32_t`: Number of columns in the matrix

---

#### rows

*Signature:* `int32_t rows() const`

Returns the number of rows in the matrix.

**Returns**

* `int32_t`: Number of rows in the matrix

---

#### size

*Signature:* `int32_t size() const`

Returns the number of rows in a vector.

For statically-typed column vectors only.

**Returns**

* `int32_t`: Number of rows in the vector

---

#### staticType

*Signature:* `constexpr MatrixStaticType staticType()`

Returns the matrix's static type.

Static types are: 0 (`MATRIX`) for general matrix, 1 (`COLUMN_VECTOR`) for column vector.

**Returns**

* `MatrixStaticType`: 0 for matrix, 1 for column vector.

---
---

### Setters

#### set (Matrix)

*Signature:* `void set(int32_t row_index, int32_t col_index, T new_value)`

Sets the value at position (`row_index`, `col_index`) to `new_value`.

Uses 0-based indexing. The element in the first row and the first column is at position (0, 0).

**Parameters**

* `row_index` (`int32_t`): Row index to set (0-based). Must satisfy `0 <= row_index < rows()`.
* `col_index` (`int32_t`): Column index to set (0-based). Must satisfy `0 <= col_index < cols()`.
* `new_value` (`T`): Value to change position (`row_index`, `col_index`) to.

---

#### set (Column Vector)

*Signature:* `void set(int32_t row_index, T new_value)`

Sets the value at position `row_index` to `new_value`.

Uses 0-based indexing. The first element is at position 0.

For statically-typed column vectors only.

**Parameters**

* `row_index` (`int32_t`): Row index to set (0-based). Must satisfy `0 <= row_index < rows()`.
* `new_value` (`T`): Value to change position `row_index` to.

---
---

### Methods

#### Constant (Matrix) Static Method

*Signature:* `static Matrix Constant(int32_t n_rows, int32_t n_cols, const T& constant_value)`

Returns a new matrix with `n_rows` rows and `n_cols` columns, all initialized to `constant_value`.

**Returns**

* `Matrix`: A new matrix with all elements set to the provided constant value.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the matrix. Must be positive.
* `n_cols` (`int32_t`): Number of columns in the matrix. Must be positive.
* `constant_value` (`T`): Value to initialize all matrix elements with.

---

#### Constant (Column Vector) Static Method

*Signature:* `static Matrix Constant(int32_t n_rows, const T& constant_value)`

Returns a statically-typed column vector with `n_rows` rows, all initialized to `constant_value`.

May be called only on statically-typed column vectors.

**Returns**

* `Matrix<T, COLUMN_VECTOR>`: A statically-typed column vector with all elements set to the provided constant value.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the vector. Must be positive.
* `constant_value` (`T`): The value to initialize all vector elements with.

---

#### cwiseProduct

*Signature:* `template<typename Tp, MatrixStaticType other_static_type> Matrix<Tp, static_type> cwiseProduct(const Matrix<Tp, other_static_type>& other) const`

Returns a new matrix containing the element-wise product of the matrix and another matrix.

**Returns**

* `Matrix<Tp, static_type>`: A new matrix with the element-wise product.

**Parameters**

* `other` (`Matrix<Tp, other_static_type>`): The matrix to multiply. Must have the same number of rows and columns as this matrix.

---

#### exp

*Signature:* `Matrix exp() const`

Returns a new matrix containing the element-wise exponential (e^x) of each element.

**Returns**

* `Matrix`: e^(each element in the matrix)

---

#### log

*Signature:* `Matrix log() const`

Returns a new matrix containing the element-wise natural logarithm (base e) of each element.

**Returns**

* `Matrix`: ln(each element in the matrix)

---

#### max

*Signature:* `Matrix max(const T& max_value) const`

Returns a new matrix where all values less than or equal to `max_value` are kept as-is, and all values greater than `max_value` are replaced by `max_value`.

**Returns**

* `Matrix`: A new matrix where all values are no greater than `max_value`.

**Parameters**

* `max_value` (`T`): Maximum allowed value in the matrix.

---

#### maxCoeff

*Signature:* `T maxCoeff() const`

Returns the maximum value stored in the matrix.

**Returns**

* `T`: Largest matrix element.

---

#### min

*Signature:* `Matrix min(const T& min_value) const`

Returns a new matrix where all values greater than or equal to `min_value` are kept as-is, and all values less than `min_value` are replaced by `min_value`.

**Returns**

* `Matrix`: A new matrix where no value is less than `min_value`.

**Parameters**

* `min_value` (`T`): Minimum allowed value in the matrix.

---

#### minCoeff

*Signature:* `T minCoeff() const`

Returns the minimum value stored in the matrix.

**Returns**

* `T`: Smallest matrix element.

---

#### Random (Matrix) Static Method

*Signature:* `static Matrix Random(int32_t n_rows, int32_t n_cols, const T& range = 1)`

Returns a new matrix of dimensions `n_rows` by `n_cols`, with elements randomly initialized within the range `[-range, range]`.

`range` is treated as its absolute value. `range` equaling -1 also gives output elements on the interval [-1, 1].


**Returns**

* `Matrix`: A matrix containing random elements within the specified range.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the matrix. Must be positive.
* `n_cols` (`int32_t`): Number of columns in the matrix. Must be positive.
* `range` (`T`): Maximum absolute value for each element. Default 1.

---

#### Random (Column Vector) Static Method

*Signature:* `static Matrix<T, COLUMN_VECTOR> Random(int32_t n_rows, VectorTag = {}, const T& range = 1)`

Returns a statically-typed column vector with `n_rows` rows, with elements randomly initialized within the range `[-range, range]`.

`range` is treated as its absolute value. `range` equaling -1 also gives output elements on the interval [-1, 1].

The `VectorTag` argument distinguishes between this method and the version for statically-typed matrices.  
Example of usage:
```
//Random 3x100 matrix initialized to random values on the interval [-1, 1]
Matrix<double, MATRIX> m = Matrix<double, MATRIX>::Random(3, 100);

//Random 3-D vector initialized to random values on the interval [-100, 100]
Matrix<double, COLUMN_VECTOR> v = Matrix<double, COLUMN_VECTOR>::Random(3, VectorTag{}, 100);
```

**Returns**

* `Matrix<T, COLUMN_VECTOR>`: A statically-typed column vector containing random elements within the specified range.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the vector. Must be positive.
* `VectorTag`: Distinguishes between calls to this method and calls to the Matrix version of `Random`.
* `range` (`T`): Maximum absolute value for each element. Default 1.

---

#### squaredNorm

*Signature:* `T squaredNorm() const`

Returns the sum of the squares of all elements in the matrix.

Equals (element 1)^2 + (element 2)^2 + ... + (element n)^2.

**Returns**

* `T`: Sum of all squared elements.

---

#### sum

*Signature:* `T sum() const`

Returns the sum of all elements in the matrix.

**Returns**

* `T`: Sum of all matrix elements.

---

#### transpose

*Signature:* `Matrix<T, MATRIX> transpose() const`

Returns the transpose (rows and columns switched) of this matrix.

Always returns a statically-typed matrix, regardless of the input's static type.

**Returns**

* `Matrix<T, MATRIX>`: A statically-typed matrix where rows and columns are swapped.

---

#### unaryExpr

*Signature:* `template<typename UnaryFunc> Matrix unaryExpr(UnaryFunc func) const`

Returns a new matrix where the function `func` is applied to each element of the matrix.

**Returns**

* `Matrix`: A new matrix where the function has been applied to each element.

**Parameters**

* `func` (`UnaryFunc`): A single-argument function, whose input is of type `T`, to apply to each matrix element.

---

#### Zero (Matrix) Static Method

*Signature:* `static Matrix Zero(int32_t n_rows, int32_t n_cols)`

Returns a matrix of dimensions `n_rows` by `n_cols`, with all elements initialized to 0.

**Returns**

* `Matrix`: A new zero matrix.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the matrix. Must be positive.
* `n_cols` (`int32_t`): Number of columns in the matrix. Must be positive.

---

#### Zero (Column Vector) Static Method

*Signature:* `static Matrix<T, COLUMN_VECTOR> Zero(int32_t n_rows)`

Returns a statically-typed column vector with `n_rows` rows, with all elements initialized to 0.

**Returns**

* `Matrix<T, COLUMN_VECTOR>`: A new zero vector.

**Parameters**

* `n_rows` (`int32_t`): Number of rows in the column vector. Must be positive.

---
---

### Arithmetic Operator Overloads

Overloads work only if the datatypes of both operands match.  
Adding matrices of types `int32_t` and `double` will not work.

#### operator+ (Matrix + Scalar)

*Signature:* `Matrix operator+(T scalar) const`

Returns a new matrix where each element equals the original matrix element plus `scalar`.

**Returns**

* `Matrix`: A new matrix with each element equal to this matrix + `scalar`.

**Parameters**

* `scalar` (`T`): Value to add.

---

#### operator+ (Scalar + Matrix)

*Signature:* `template<typename T, MatrixStaticType static_type> Matrix<T, static_type> operator+(const T& scalar, const Matrix<T, static_type>& matrix)`

Returns a new matrix where each element of the matrix is added to `scalar`.

The scalar appears on the left-hand side of the operation.

Works for any matrix static type.

**Returns**

* `Matrix<T, static_type>`: A new matrix where each element equals `scalar` + original.

**Parameters**

* `scalar` (`T`): Value to add to each matrix element.
* `matrix` (`Matrix<T, static_type>`): Matrix to add to (of any static type).

---

#### operator+ (Matrix + Matrix)

*Signature:* `Matrix operator+(const Matrix& other) const`

Returns a new matrix containing the element-wise sum of the current matrix and another matrix.

Works regardless of static type, as long as both matrices' row and column counts match.

**Returns**

* `Matrix`: A new matrix with the coefficient-wise sum.

**Parameters**

* `other` (`Matrix`): Matrix to add. Must match this matrix's row and column counts.

---

#### operator+= (Scalar Add-Assign)

*Signature:* `void operator+=(const T& scalar)`

Adds `scalar` to the current matrix, modifying the matrix in place.

**Parameters**

* `scalar` (`const T&`): Value to add to the matrix.

---

#### operator+= (Matrix Add-Assign)

*Signature:* `template<typename Tp, MatrixStaticType other_static_type> void operator+=(const Matrix<Tp, other_static_type>& other)`

Adds `other` to the current matrix, modifying the matrix in place.

Works for any matrix static type, as long as the matrices' row and column counts are the same.

**Parameters**

* `other` (`Matrix<Tp, other_static_type>`): The matrix to add to the current matrix. Must match this matrix's row and column counts (although static type may differ).

---

#### operator- (Matrix - Scalar)

*Signature:* `Matrix operator-(T scalar) const`

Returns a new matrix where each element equals the original matrix element minus `scalar`.

**Returns**

* `Matrix`: A new matrix with each element equal to original - `scalar`.

**Parameters**

* `scalar` (`T`): Scalar value to subtract.

---

#### operator- (Scalar - Matrix)

*Signature:* `template<typename T, MatrixStaticType static_type> Matrix<T, static_type> operator-(const T& scalar, const Matrix<T, static_type>& matrix)`

Returns a new matrix where each element of the matrix is subtracted from `scalar`.

The scalar appears on the left-hand side of the operation.

Works for any matrix static type.

**Returns**

* `Matrix`: A new matrix where each element equals `scalar` - original.

**Parameters**

* `scalar` (`T`): The scalar value to subtract from each matrix element.
* `matrix` (`Matrix<T, static_type>`): The matrix to subtract from.

---

#### operator- (Matrix - Matrix)

*Signature:* `Matrix operator-(const Matrix& other) const`

Returns a new matrix containing the element-wise difference between the current matrix and another matrix.

This operation works regardless of `other`'s static type, as long as row and column counts match.

**Returns**

* `Matrix`: A new matrix with the coefficient-wise difference.

**Parameters**

* `other` (`Matrix`): Matrix to subtract. Must have the same number of rows and columns as this matrix.

---

#### operator-= (Scalar Subtract-Assign)

*Signature:* `void operator-=(const T& scalar)`

Subtracts `scalar` from the current matrix, modifying the matrix in place.

**Parameters**

* `scalar` (`const T&`): Value to subtract from the matrix.

---

#### operator-= (Matrix Subtract-Assign)

*Signature:* `template<typename Tp, MatrixStaticType other_static_type> void operator-=(const Matrix<Tp, other_static_type>& other)`

Subtracts `other` from the current matrix, modifying the matrix in place.

This operation works regardless of `other`'s static type, as long as row and column counts match.

**Parameters**

* `other` (`Matrix<Tp, other_static_type>`): The matrix to subtract from the current matrix.

---

#### operator* (Matrix * Scalar)

*Signature:* `Matrix operator*(const T& scalar) const`

Returns a new matrix where each element equals the original matrix element times `scalar`.

**Returns**

* `Matrix`: A new matrix with each element equal to `scalar` * original value.

**Parameters**

* `scalar` (`T`): The scalar value to multiply by.

---

#### operator* (Scalar * Matrix)

*Signature:* `template<typename T, MatrixStaticType static_type> inline Matrix<T, static_type> operator*(const T& scalar, const Matrix<T, static_type>& matrix)`

Returns a new matrix where each element is the result of multiplying the matrix element by `scalar`.

The scalar is on the left-hand side of the operation.

**Returns**

* `Matrix`: A new matrix where each element equals `scalar` * matrix.

**Parameters**

* `scalar` (`T`): The scalar value to multiply by.
* `matrix` (`Matrix<T, static_type>`): The matrix to multiply.

---

#### operator* (Matrix Multiplication)

*Signature:* `template<typename T, MatrixStaticType lhs_static_type, MatrixStaticType rhs_static_type> inline Matrix<T, MATRIX> operator*(const Matrix<T, lhs_static_type>& lhs, const Matrix<T, rhs_static_type>& rhs)`

Returns the matrix product of the given matrices.

This operation works regardless of the operands' static type, as long as row and column counts are compatible.

ATTENTION! Unlike standard Eigen, Eigen Lite does not support implicit element-wise multiplication.  
For element-wise multiplication, use the `cwiseProduct` method: `lhs.cwiseProduct(rhs)`.

**Returns**

* `Matrix`: A new matrix, of dimension `lhs.rows()` by `rhs.cols()`, containing the result of the matrix multiplication.

**Parameters**

* `lhs` (`Matrix<T, lhs_static_type>`): Left-hand matrix operand.
* `rhs` (`Matrix<T, rhs_static_type>`): Right-hand matrix operand. Must satisfy `lhs.cols() == rhs.rows()`.

---

#### operator*= (Scalar Multiply-Assign)

*Signature:* `void operator*=(const T& scalar)`

Multiplies `scalar` by each value in the current matrix, modifying the matrix in place.

**Parameters**

* `scalar` (`const T&`): Scalar value to multiply by the matrix.

---

#### operator/ (Matrix / Scalar)

*Signature:* `Matrix operator/(const T& scalar) const`

Returns a new matrix where each element equals the original matrix element divided by `scalar`.

**Returns**

* `Matrix`: A new matrix with each element equal to original / `scalar`.

**Parameters**

* `scalar` (`T`): Scalar divisor. Cannot be zero.

---

#### operator/= (Scalar Divide-Assign)

*Signature:* `void operator/=(const T& scalar)`

Divides each element of the matrix by `scalar`, modifying the matrix in place.

**Parameters**

* `scalar` (`T`): Scalar value to divide by. Cannot be zero.

---
---

### Other Operator Overloads

#### operator() (Matrix, Mutable)

*Signature:* `T& operator() (int32_t row_index, int32_t col_index)`

Returns a reference to the matrix element at position `(row_index, col_index)`.

Allows for index access.

Value indexing is 0-based. The top left value is at position (0, 0).

Example
```
Given a matrix `m` containing:
[[ 1  2  3 ]
 [ 4  5  6 ]]

std::cout << m(0, 0) << std::endl; // prints 1
std::cout << m(0, 1) << std::endl; // prints 2
std::cout << m(1, 2) << std::endl; // prints 6
```

**Returns**

* `T&`: Reference to the matrix element at position `(row_index, col_index)`.

**Parameters**

* `row_index` (`int32_t`): Row index. Must satisfy `0 <= row_index < rows()`.
* `col_index` (`int32_t`): Column index. Must satisfy `0 <= col_index < cols()`.

---

#### operator() (Matrix, Const)

*Signature:* `const T& operator() (int32_t row_index, int32_t col_index) const`

Returns a constant reference to the matrix element at position `(row_index, col_index)`.

Allows for index access.

Value indexing is 0-based. The top left value is at position (0, 0).

Example
```
Given a matrix `m`, which may be const, containing:
[[ 1  2  3 ]
 [ 4  5  6 ]]

std::cout << m(0, 0) << std::endl; // prints 1
std::cout << m(0, 1) << std::endl; // prints 2
std::cout << m(1, 2) << std::endl; // prints 6
```

**Returns**

* `const T&`: Constant reference to the matrix element at position `(row_index, col_index)`.

**Parameters**

* `row_index` (`int32_t`): The row index. Must satisfy `0 <= row_index < rows()`.
* `col_index` (`int32_t`): The column index. Must satisfy `0 <= col_index < cols()`.

---

#### operator() (Column Vector, Mutable)

*Signature:* `T& operator() (int32_t row_index)`

Returns a reference to the element at position `row_index`.

Allows for index access.

Value indexing is 0-based. The top value is at position 0.

For statically-typed column vectors only.

Example
```
Given a statically-typed column vector `v` containing:
[[ 1 ]
 [ 2 ]
 [ 3 ]]

std::cout << v(0) << std::endl; // prints 1
std::cout << v(2) << std::endl; // prints 3
```

**Returns**

* `T&`: Reference to the matrix element at position `row_index` in a column vector.

**Parameters**

* `row_index` (`int32_t`): The row index. Must satisfy `0 <= row_index < rows()`.

---

#### operator() (Column Vector, Const)

*Signature:* `const T& operator() (int32_t row_index) const`

Returns a constant reference to the element at position `row_index`.

Allows for index access.

Value indexing is 0-based. The top value is at position 0.

For statically-typed column vectors only.

Example
```
Given a statically-typed column vector `v`, which may be const, containing:
[[ 1 ]
 [ 2 ]
 [ 3 ]]

std::cout << v(0) << std::endl; // prints 1
std::cout << v(2) << std::endl; // prints 3
```

**Returns**

* `const T&`: Constant reference to the matrix element at position `row_index` in a column vector.

**Parameters**

* `row_index` (`int32_t`): The row index. Must satisfy `0 <= row_index < rows()`.

---

#### operator<< (Comma Initializer)

*Signature:* `template<typename T> CommaInitializer operator<<(T value)`

Initializes this matrix with the given comma-separated values, returning an internal `CommaInitializer` object.

Values are loaded in row-major order.

Usage example for a 2x3 matrix:
```
using namespace Eigen;
Matrix<double, MATRIX> m(2, 3);
m << 1, 2, 3, 4, 5, 6;
std::cout << m;
```
Example's output:
```
[[ 1  2  3 ]
 [ 4  5  6 ]]
```

**Returns**

* `CommaInitializer`: Internal object for initializing the matrix with a comma-separated list.

**Parameters**

* `value` (`T`): Comma-separated list, of type `T`, to initialize the matrix with. Must have length `rows()` * `cols()`.

---

#### operator<< (Output Stream Insertion)

*Signature:* `template<typename CharT, typename Traits, typename Tp, MatrixStaticType m_static_type>  std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Matrix<Tp, m_static_type>& matr)`

Exports the matrix to `output_stream`, returning a reference to `output_stream` with `matr` inside.

Allows for printing matrices to `std::cout` and related output streams.

Each row is printed on a new line.

**Returns**

* `std::basic_ostream<CharT, Traits>&`: Output stream with the matrix inserted.

**Parameters**

* `output_stream` (`std::basic_ostream<CharT, Traits>`): Output stream to write to.
* `matr` (`Matrix<Tp, m_static_type>`): Matrix to export to the output stream.

---
---
---
[Back to table of contents](#table-of-contents)