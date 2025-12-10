#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the Python executable
PYTHON_EXEC="$SCRIPT_DIR/tf_env/bin/python"

echo "--- Starting MLP Iteration Sequence ---"
echo "Script directory: $SCRIPT_DIR"
echo "Python executable: $PYTHON_EXEC"
echo " "

# --- Iteration 1: Baseline MLP (Two Hidden Layers) ---
echo "## Running Iteration 1: Baseline MLP"
$PYTHON_EXEC "$SCRIPT_DIR/HeartDieaseMLP.py"
echo " "

# --- Iteration 2: 2-Layer MLP with 5-Fold Cross-Validation ---
echo "## Running Iteration 2: 2-Layer MLP with CV"
$PYTHON_EXEC "$SCRIPT_DIR/MLP_test.py"
echo " "

# --- Iteration 3: 2-Layer MLP with Ensemble Bagging ---
echo "## Running Iteration 3: 2-Layer MLP with Ensemble Bagging"
$PYTHON_EXEC "$SCRIPT_DIR/MLP_Test2.py"
echo " "

# --- Iteration 4: 2-Layer MLP without F1 Callback (Clean) ---
echo "## Running Iteration 4: 2-Layer MLP without F1 Callback"
$PYTHON_EXEC "$SCRIPT_DIR/MLP_test3.py"
echo " "

# --- Iteration 5: Recall Optimized/Fixed HPs ---
echo "## Running Iteration 5: Recall Optimized/Fixed HPs"
$PYTHON_EXEC "$SCRIPT_DIR/MLP_test4.py"
echo " "

# --- Iteration 6: 2-Layer MLP with SMOTE-Tomek 10-Fold CV ---
echo "## Running Iteration 6: 2-Layer MLP with SMOTE-Tomek 10-Fold CV"
$PYTHON_EXEC "$SCRIPT_DIR/MLP_test5.py"
echo " "

# --- Iteration 7: Single-Layer MLP 5-Fold SMOTETomek CV ---
echo "## Running Iteration 7: Single Layer MLP 5-Fold CV"
$PYTHON_EXEC "$SCRIPT_DIR/MLP_Test6_oneHiddenlayer.py"
echo " "

echo "--- All MLP iterations complete! ---"