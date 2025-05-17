"""
Utility functions and classes for the evaluation scripts.
"""

import sys
import datetime

class OutputCapture:
    """
    Utility class for capturing output to both console and file.
    
    This class provides a way to print output to both the console and a file
    simultaneously, making it easy to save evaluation results while still
    displaying them to the user.
    """
    def __init__(self, output_file=None):
        """
        Initialize the OutputCapture with an optional output file.
        
        Args:
            output_file (str, optional): Path to the file where output should be saved.
                                         If None, output is only printed to console.
        """
        self.output_file = output_file
        self.file = None
        
        if output_file:
            try:
                self.file = open(output_file, 'w', encoding='utf-8')
                print(f"Saving output to {output_file}")
                
                # Save command and parameters at the top of the file
                self._save_command_info()
            except Exception as e:
                print(f"Warning: Could not open output file {output_file}: {e}")
                self.file = None
    
    def _save_command_info(self):
        """Save the command and parameters used to run the script."""
        if self.file:
            # Get current date and time
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get the command line arguments
            command = " ".join(sys.argv)
            
            # Write to file
            self.file.write(f"# Evaluation run on {date_str}\n")
            self.file.write(f"# Command: {command}\n\n")
            self.file.flush()
    
    def print(self, *args, **kwargs):
        """
        Print to both console and file if available.
        
        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments to be passed to print().
        """
        # Print to console
        print(*args, **kwargs)
        
        # Also write to file if available
        if self.file:
            # Convert all arguments to strings and join with spaces
            output_str = ' '.join(str(arg) for arg in args)
            
            # Handle the 'end' parameter if provided in kwargs
            end = kwargs.get('end', '\n')
            
            # Write to file
            self.file.write(output_str + end)
            self.file.flush()  # Ensure it's written immediately
    
    def close(self):
        """Close the output file if it was opened."""
        if self.file:
            self.file.close()
            self.file = None
