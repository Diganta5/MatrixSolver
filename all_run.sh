#!/bin/bash

# Run the Python script
python omeg.py

# Check if the Python script was successful
if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
    
    # Compile and run the C program
    gcc Matsolve.c -o abc
    if [ $? -eq 0 ]; then
        echo "C program compiled successfully."
        ./abc
        if [ $? -eq 0 ]; then
            echo "C program executed successfully."
        else
            echo "C program execution failed."
        fi
    else
        echo "C program compilation failed."
    fi
else
    echo "Python script execution failed."
fi
