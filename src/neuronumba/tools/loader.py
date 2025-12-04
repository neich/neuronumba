import os

import numpy as np

import neuronumba.tools.hdf as hdf

import numpy as np
import re

def read_csv_with_repeated_delimiters(file_path, delimiter=None, encoding='utf-8', dtype=float, skip_header=False):
    """
    Read a CSV file containing numeric values with automatic delimiter inference.

    This function handles CSV files where delimiters may appear consecutively multiple times.
    It automatically infers the delimiter if not specified and enforces that all values
    must be numeric (no blank values allowed within rows).

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    delimiter : str, optional
        The delimiter character. If None, will be automatically inferred from common
        delimiters: comma, semicolon, tab, space, pipe
    encoding : str, default 'utf-8'
        File encoding
    dtype : data type, default float
        Data type for the resulting array (must be numeric: int, float, np.float64, etc.)
    skip_header : bool, default False
        Whether to skip the first row (assumed to be header)

    Returns:
    --------
    numpy.ndarray
        The data as a numpy array with the specified numeric dtype

    Raises:
    -------
    ValueError
        If non-numeric values are found, if rows have blank values, or if delimiter
        cannot be inferred

    Examples:
    ---------
    >>> # File with repeated commas: "1,,2,,,3\\n4,,,5,,6"
    >>> data = read_csv_with_repeated_delimiters('data.csv')
    >>> print(data)
    [[1. 2. 3.]
     [4. 5. 6.]]

    >>> # Specify delimiter explicitly
    >>> data = read_csv_with_repeated_delimiters('data.tsv', delimiter='\\t')

    >>> # Get integer array
    >>> data = read_csv_with_repeated_delimiters('integers.csv', dtype=int)
    """
    
    COMMON_DELIMITERS = [',', ';', '\t', ' ', '|']

    def is_numeric(value):
        """Check if a string value can be converted to a number."""
        try:
            float(value.strip())
            return True
        except (ValueError, AttributeError):
            return False

    def split_line_with_delimiter(line, delim):
        """Split a line using the given delimiter, collapsing repeated delimiters."""
        if not line.strip():
            return []

        line = line.strip()

        # Replace multiple consecutive delimiters with single delimiter
        escaped_delimiter = re.escape(delim)
        cleaned_line = re.sub(f"{escaped_delimiter}+", delim, line)

        # Remove leading/trailing delimiters
        cleaned_line = cleaned_line.strip(delim)

        if cleaned_line:
            fields = [field.strip() for field in cleaned_line.split(delim)]
            return fields
        return []

    def infer_delimiter_for_line(line):
        """
        Infer the best delimiter for a single line by trying common delimiters
        and returning the one that produces the most numeric fields.
        """
        best_delimiter = None
        best_count = 0

        for delim in COMMON_DELIMITERS:
            fields = split_line_with_delimiter(line, delim)
            if fields and all(is_numeric(f) for f in fields):
                if len(fields) > best_count:
                    best_count = len(fields)
                    best_delimiter = delim

        return best_delimiter, best_count

    def infer_delimiter_for_file(lines, start_line):
        """
        Infer the delimiter for the entire file by checking which delimiter
        works consistently across all data lines.
        """
        # First, try to infer from the first data line
        for line_num, line in enumerate(lines):
            if line_num < start_line:
                continue
            if not line.strip():
                continue
            
            best_delim, _ = infer_delimiter_for_line(line)
            if best_delim is not None:
                # Verify this delimiter works for all lines
                all_valid = True
                num_cols = None
                
                for check_line_num, check_line in enumerate(lines):
                    if check_line_num < start_line:
                        continue
                    if not check_line.strip():
                        continue
                    
                    fields = split_line_with_delimiter(check_line, best_delim)
                    if not fields or not all(is_numeric(f) for f in fields):
                        all_valid = False
                        break
                    
                    if num_cols is None:
                        num_cols = len(fields)
                    elif len(fields) != num_cols:
                        all_valid = False
                        break
                
                if all_valid:
                    return best_delim
        
        # If first line's delimiter doesn't work for all, try all delimiters
        for delim in COMMON_DELIMITERS:
            all_valid = True
            num_cols = None
            
            for line_num, line in enumerate(lines):
                if line_num < start_line:
                    continue
                if not line.strip():
                    continue
                
                fields = split_line_with_delimiter(line, delim)
                if not fields or not all(is_numeric(f) for f in fields):
                    all_valid = False
                    break
                
                if num_cols is None:
                    num_cols = len(fields)
                elif len(fields) != num_cols:
                    all_valid = False
                    break
            
            if all_valid and num_cols is not None and num_cols > 0:
                return delim
        
        return None

    # Main processing logic
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            lines = file.readlines()

        if not lines:
            raise ValueError("File is empty")

        start_line = 1 if skip_header else 0

        # Infer delimiter if not provided
        if delimiter is None:
            delimiter = infer_delimiter_for_file(lines, start_line)
            if delimiter is None:
                raise ValueError(
                    "Could not infer delimiter. File may not contain valid numeric data "
                    "or may use an unsupported delimiter. Try specifying delimiter explicitly."
                )

        # Now parse all lines using the inferred/provided delimiter
        data_rows = []
        num_cols = None

        for line_num, line in enumerate(lines):
            if line_num < start_line:
                continue
            if not line.strip():
                continue

            fields = split_line_with_delimiter(line, delimiter)
            if not fields:
                continue

            # Check all fields are numeric
            for col_idx, field in enumerate(fields):
                if not is_numeric(field):
                    raise ValueError(
                        f"Non-numeric value '{field}' found at row {line_num + 1}, column {col_idx + 1}"
                    )

            # Check consistent column count
            if num_cols is None:
                num_cols = len(fields)
            elif len(fields) != num_cols:
                raise ValueError(
                    f"Inconsistent column count at row {line_num + 1}: expected {num_cols}, got {len(fields)}"
                )

            data_rows.append(fields)

        if not data_rows:
            raise ValueError("No valid data rows found")

        # Convert to numeric values
        processed_data = []
        for row_idx, row in enumerate(data_rows):
            processed_row = []
            for col_idx, value in enumerate(row):
                try:
                    if dtype in (int, np.int32, np.int64):
                        processed_row.append(int(float(value.strip())))
                    else:
                        processed_row.append(float(value.strip()))
                except ValueError as e:
                    raise ValueError(
                        f"Non-numeric value '{value}' found at row {row_idx + 1}, column {col_idx + 1}"
                    ) from e
            processed_data.append(processed_row)

        return np.array(processed_data, dtype=dtype)

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        raise
    except Exception as e:
        raise Exception(f"Error processing file {file_path}: {str(e)}")




def load_2d_matrix(filename, delimiter=None, index=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.csv':
        return read_csv_with_repeated_delimiters(filename, delimiter=delimiter)
    elif file_extension == '.tsv':
        # For .tsv files, default to tab if no delimiter specified
        if delimiter is None:
            delimiter = '\t'
        return read_csv_with_repeated_delimiters(filename, delimiter=delimiter)
    elif file_extension == '.txt':
        # For .txt files, auto-infer delimiter
        return read_csv_with_repeated_delimiters(filename, delimiter=delimiter)
    elif file_extension == '.mat':
        if index is None:
            raise RuntimeError("You have to provide an index for the file")
        return hdf.loadmat(filename)[index]
    elif file_extension == '.npy':
        return np.load(filename)
    elif file_extension == '.npz':
        return np.load(filename, allow_pickle=True)[index]
    else:
        raise RuntimeError("Unrecognized file extension")