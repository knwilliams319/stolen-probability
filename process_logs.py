'''
This script is used to process the log files outputted by the `fairseq-train` CLI tool.
These files contain lots of duplicate lines, so we can save space on the disk by 
post-processing them. Find how to use this script at the command line via:

python process_logs.py -h
'''
from pathlib import Path
import argparse
from sys import argv

# SECTION: Initialize the argparse object and process arguments
parser = argparse.ArgumentParser(description='Clean up a logfile by removing duplicate lines')
parser.add_argument('filepath', 
                    type=Path,
                    help='Path to the logfile to be processed')
parser.add_argument('-d', '--delete', 
                    action='store_true',
                    default=False,
                    help='Whether the original logfile should be deleted after being processed')

args = parser.parse_args(argv[1:])  # argv[0] is always 'process_logs.py'
log_path = args.filepath
should_delete = args.delete
#!SECTION

# SECTION: Perform the processing
if not log_path.exists():
    raise ValueError(f'The path {log_path} does not exist!')

output_path = Path(log_path.parent, f'processed_{log_path.name}')  # write non-duplicates to new file
seen_lines = set()  # used to track duplicates

start_after_line = 'begin dry-run validation on "test" subset\n' # until we see this line, we will not remove duplicates
start_after_line_seen = False

with output_path.open(mode='w') as out_file:
      with log_path.open(mode='r') as in_file:
           for line in in_file:
                if start_after_line_seen:
                     if line not in seen_lines:
                          out_file.write(line)
                          seen_lines.add(line)
                     if line.startswith('end of epoch'): 
                          # we will never see a duplicate line after this signal, so we can reset 
                          # the seen_lines set to save memory
                          seen_lines = set()
                else:
                     out_file.write(line)
                     if line == start_after_line:
                          start_after_line_seen = True

if should_delete:
     log_path.unlink()
#!SECTION
