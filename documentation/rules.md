# PROJECT RULES

[Back to README](../README.md)

This is a project of rules and regulations! Read this document thoroughly before
starting work on the project!
As a friendly reminder, failure to follow the rules means I will hunt you down and [DATA EXPUNGED].

I reserve the rights to change, update, nullify, or interpret any of the rules at any time, for any reason.

### Table of contents
- [Docstring Rules](#docstring-rules)
- [Method Rules](#method-rules)
- [Inheritance Rules](#inheritance-rules)
- [Naming Rules](#naming-rules)
- [Class Structure Rules](#class-structure-rules)
- [Final Remarks](#final-remarks)




## Docstring Rules
All docstrings must follow the template given below:  

**BEGIN FORMAT**

One-line summary of the method or class.  
If the method's purpose is to return a value, the summary must begin with "Returns..."  
If the method modifies any of its inputs, the modifications should be stated here.

Additional details of the method or class. This may span multiple lines.  
If each line significantly exceeds ~80 characters (the exact meaning of "significantly" is up to the programmer's judgment),
any additional text must be on a new line. 

One or two lines of important notice for clients, i.e. preconditions not enforceable by assertions, or details that are easy to overlook.  
This section must contain "Helper to {parent method name}" if the method is a private helper to another method.

@param param1 the first parameter, followed by a period. Then state any preconditions on the parameter, followed by a period. If the parameter has a default value, state "Default: {default value}".  
@param param2...  
@return some value. Return value descriptions are required! Must come immediately after the parameters   
@throws some_exception...*  
@throws some_other_exception...*  

*Exceptions thrown must be listed in alphabetical order.

**END FORMAT**<br><br>

Short form docstring for getter methods:

**BEGIN FORMAT**

@return what the getter method returns, followed by a period.

On a new line, give extra information about the method.

**END FORMAT**<br><br>


NOTE: All parameter, return value, and exception entries must have a description, even if the entries are trivial or redundant!  
All fields in *any* class or struct must also have docstrings! These docstrings can be a one-line summary.

Clarity is more important than strict grammatical adherence.

Class docstrings need not include parameters, return values, or exception information.


**Sample Docstring**
```
/**
 * Returns `input` with all indices at and after `start_index` divided by 2.
 * 
 * This method cannot be called if the class attribute `reference_count` 
 * is negative (throws `illegal_state` if so).
 * 
 * @param input input vector to halve
 * @param start_index index to begin division at. Cannot be negative. Default: 0
 * @return copy of `input` with indices halved, except for the last `start_index` indices
 * @throws `illegal_state` if the method is called when `reference_count` is negative
 */
VectorXd halve_to_index(const VectorXd& input, int start_index = 0) {
    ...
}
```


## Method Rules

Any modification to a method's input must occur on a deep defensive copy. Passing a non-constant reference to a method is not allowed, unless otherwise specified.  
Examples:  
OK- `void method(const Object& obj)`  
OK- `void method(Object obj)`  
NOT OK, unless specified- `void method(Object& obj)`


All methods must assert their preconditions. Descriptive messages should be used in assertions.  
Example:  
`assert((index>=0 && "Index cannot be negative"));`

Any method that takes a pointer type must use smart pointers only. Raw pointers as method parameters are not allowed.  
This rule applies to getters and setters, so any internal pointers must either be smart pointers or be inaccessible to outside users.

## Inheritance Rules
Multiple inheritance of concrete classes is not allowed!

A class may inherit from, at most, one concrete class.  
Any other classes inherited must be purely abstract. A purely abstract class has no fields, and all its methods are purely virtual.

## Naming Rules

Class names should use camel case, except that the first letter is capitalized. There should be no spaces between words, where each word's first letter is uppercase.  
Examples: `MyClass`, `HelloWorldPrinter`, `Calculator`

Field names must use snake case. In snake case, all letters must be lowercase. Spaces between words should be separated by underscores.  
Examples: `my_method_1`, `do_something`, `display`

Method names must use snake case.

Getter methods (methods that return the value of a private field) should be the field's name, or at least very similar to the field name.  
If the getter accesses a field by index (an identifier of integer type) in an indexed sequence (an ordered collection of objects, whose elements are uniquely identified by index numbers), the setter name must end with "at".  
Examples:  
- `value()`, to retrieve the field `value`  
- `sequence_element_at(int index)`, to retrieve a sequence element at index `index`

Setter methods (methods that change the value of a private field) should start with the word "set".  
If the setter appends a field to the end of a sequence, the setter name may start with "add".  
If the setter accesses a field by index (an identifier of integer type) in an indexed sequence (an ordered collection of objects, whose elements are uniquely identified by index numbers), the setter name must end with "at".  
Examples:  
- `set_value(Value new_value)`, to change the private field `value`  
- `add_element(Element new_element)`, to append `new_element` to the end of an indexed sequence
- `set_element_at(int index, Element new_element)`, to change the element at `index` to `new_element` in an indexed sequence  


Class names and method parameter names should use no abbreviations. Example: Choose `update_parameters(const Object& object)` instead of `update_params(const Object& obj)`.  
You may use abbreviations if the full name will make the class unwieldy or confusing.

In case of naming conflicts between parameters and class fields, the class field should take the less verbose name.




## Class Structure Rules

Classes must match the template below.  

There are lines of comments separating the different sections. The lines should appear in the class definition.
The number of comment lines may vary. Longer classes should have more comment lines separating each section.  
Comment lines are not necessary if the class is short. A good measure is if at least half of the class definition
can fit on your screen.

All methods must conform to the [Method Rules](#method-rules).

Underneath each row of comment lines is a section header. The section header should also appear in the class definition.

A class does not need to have all the listed sections. Omit any sections that a class doesn't have.

The `public` and `private` markers must be at the same level of indentation as the outside of the class.

**BEGIN CLASS FORMAT**

Docstring that conforms to the [Docstring Rules](#docstring-rules)   
NOTE: Since class definitions have no inputs or return values, 
a class docstring does not include any parameters, return values, or exceptions.

Constants, whose names are in all caps, with a descriptive name.
Must be declared in a logical order. If no logical order exists, declare in alphabetical order by variable name.  
*Docstrings must be included.*  
Example:  
```
/**
 * The error message to use if none is provided.
 */
const std::string DEFAULT_ERROR_MESSAGE = "error";

/**
 * Maximum allowed layers in a network.
 */
const int MAX_LAYERS = 1000;
```

Constants can have public access.

//////////////////////////////////////////////////////////////  

Variable fields  
Must be declared in a logical order. If no logical order exists, declare in alphabetical order by variable name.

All variable fields must be private.

Reminder: All constant and variable fields MUST HAVE DOCSTRINGS!


//////////////////////////////////////////////////////////////  
//CONSTRUCTOR

Class constructor(s). If the class has multiple constructors, 
the most frequently used constructors are first.  
ALL CLASSES must have an explicitly defined constructor, even if the constructor does nothing.
Exception: subclasses do not need explicitly defined constructors if the superclass has a defined constructor.


//////////////////////////////////////////////////////////////  
//GETTERS

Getter methods (methods that retrieve a private field), in alphabetical order by method name


//////////////////////////////////////////////////////////////  
//SETTERS

Setter methods (methods that change the value of a private field), in alphabetical order by method name


//////////////////////////////////////////////////////////////  
//METHODS (alternative title: ADDITIONAL METHODS)

Additional methods  
All additional methods should be in alphabetical order by the method's name.

Private helper methods for a single method may go immediately above the parent method.
They may break the alphabetical order rule.  
These methods must have a message in their docstrings that says "Helper to {parent method name}"

All methods must be separated by 2 or 3 blank lines.    
Whether to separate by 2 or 3 blank lines is the writer's choice,
unless the class has more than 10 methods (here, the constructor counts as a method). If so, 
methods must be separated by 3 blank lines.  
Getters and setters of the same field, methods for unit tests, or overloads of the same method
may be separated by 1 or 2 blank lines.

If a method is overloaded, the method that takes the least parameters should be first. In case of a tie, the method that takes compound objects (as opposed to primitive types) should go last. If there is still a tie, the method author may decide the order.  
Method overloads are ideally separated by 1-2 blank lines.

**END CLASS FORMAT**


Example class structure
```
/**
 * Sample class to illustrate good class organization.
 * 
 * The class can be used a finite number of times.
 * If used more than its limit, the class' error message is set
 * and the class cannot be used anymore.
 */
class MyClass {

private:

    /**
     * Error message if an error condition is ever met.
     * If there are no errors, this is the empty string.
     */
    string error_status;

    /**
     * Maximum amount of times that this object can be used.
     * Cannot be negative.
     */
    int max_usage_count;

    /**
     * Tracks the number of times 
     * that this object has been used. 
     * Cannot be negative.
     */
    int usage_count;

    

public:

    //////////////////////////////////////////////////
    //CONSTRUCTOR

    /**
     * Creates a new instance of MyClass with no tracked usages.
     * It has 100 max usages.
     */
    MyClass() {
        error_status = "";
        max_usage_count = 100;
        usage_count = 0;
    }

    /**
     * Creates a new instance of MyClass with no tracked usages.
     * Its max amount of usages starts at `initial_max_usage_count`.
     * 
     * @param initial_max_usage_count starting value for number of uses. Cannot be negative.
     */
    MyClass(int initial_max_usage_count) {
        assert((initial_usage_count>=0 && "Max initial use count cannot be negative"));
        error_status = "";
        max_usage_count = initial_max_usage_count;
        usage_count = 0;
    }


    ////////////////////////////////////////////////////////////
    //GETTERS

    /**
     * @return the object's error status.
     * 
     * If there are no errors, returns the empty string.
     */
    string error_status() {
        return error_status;
    }    


    /**
     * @return this object's maximum number of usages
     */
    int max_usage_count() {
        return max_usage_count;
    }


    /**
     * @return this object's usage count
     */
    int usage_count() {
        return usage_count;
    }


    ////////////////////////////////////////////////////////////
    //SETTERS

    /**
     * Sets the usage count to `new_count`.
     * @param new_count new usage count to set. Cannot be negative.
     */
    void set_usage_count(int new_count) {
        assert((new_count>=0 && "New usage count cannot be negative"));
        usage_count = new_count;
    }


    ////////////////////////////////////////////////////////////
    //METHODS

    /**
     * Uses this object, increasing its usage count by 1.
     *
     * If the object is used more than `max_usage_count()` times,
     * the error status is updated to "too_many_uses".
     */
    void use() {
        if(usage_count > max_usage_count) {
            error_status = "too_many_uses";
            return;
        }

        usage_count++;
    }
};
```


## Final Remarks
No contributors or viewers may make references to K-Pop Demon Hunters.

If you disagree with any of the rules, click [here](https://www.youtube.com/watch?v=xvFZjo5PgG0) to file a complaint.

[Back to table of contents](#table-of-contents)