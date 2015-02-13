# The MIT License (MIT)
# Copyright (c) 2015 Thoughtly, Corp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.



import os
from os import listdir
from os.path import isfile, join, split

# We output files as CSVs where appropriate
import unicodecsv as csv


def add_filesystem_path_args(parser, short_name, long_name, help, required, group=None):
    if group != None:
        parser = group
        
    parser.add_argument(short_name, long_name, help=help, required=required)
    
    
def directory_file_names(path, recursive, exceptions):

    if(isfile(path) == True):
        return []
    else:
        files = [join(path, f) for f in listdir(path) if (isfile(join(path,f)) and is_visible_file(f)) and is_allowed(f, exceptions) ]
    
        if recursive:
            directories = []
            for directory_or_file_name in listdir(path):
                directory_or_file_path = join(path, directory_or_file_name)
                if(isfile(directory_or_file_path) == False and is_visible_file(directory_or_file_name)):
                    directories.append(directory_or_file_name)
                    
            for directory_name in directories:
                directory_files = directory_file_names(join(path, directory_name), recursive, exceptions)
                files.extend(directory_files)

        return files
        
def is_allowed(file_name, exceptions):
    if(exceptions == None):
        return True
    else:
        return file_name not in exceptions
    
def path_with_new_root_directory(file_name, input_directory, output_directory):    

    path = file_name.split(os.sep)
    input_directory_split = None
    if(input_directory):
        input_directory_split = input_directory.split(os.sep)
    output_directory_split = output_directory.split(os.sep)
    
    found_end_of_input=False
    for index,value in enumerate(path):
        if(input_directory_split == None or index >= len(input_directory_split) or found_end_of_input):
            output_directory_split.append(value)
            
        elif found_end_of_input == False and value != input_directory_split[index]:
            found_end_of_input = True
            output_directory_split.append(value)            
#            print ("found end at " + str(index) + " " + value)

#    print "output: " + os.sep.join(output_directory_split)
    return os.sep.join(output_directory_split)

def input_directory(args):
    input_directory = args["input"]
    if isfile(input_directory):
        input_directory = ""

    return input_directory
    
def open_output_file(file_name, input_directory, output_directory, append, default_fd, extension=None):
    out_file = None
    if(output_directory):
        out_file_name = path_with_new_root_directory(file_name, input_directory, output_directory)
        if(append == False or os.path.exists(out_file_name) == False):
            out_file = create_for_write(swap_extension(out_file_name, extension))
    else:
        out_file = default_fd
    return out_file
    
def open_input_file(file_name, input_directory):
    return open(join(input_directory, file_name))
    
def create_for_write(out_file_name):
    path = split(out_file_name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
        
    out_file = open(out_file_name, 'w+')
    return out_file

def is_visible_file(f):
    return f[0] != '.'
    
def swap_extension(file, extension=None):
    if(extension == None):
        return file
        
    no_extension = os.path.splitext(file)[0]
    no_extension = no_extension + "." + extension
    return no_extension

def remove_numbering(name_in):
    tokens = name_in.split(".")
    if(tokens.count >= 3):
        number = tokens[-2]
        del tokens[-2]
        name = ".".join(tokens)
        return name
        
    return None
    

def file_names_at_path(path, recursive, exceptions=None):
    if(isfile(path)):
        file_names = [path]
    else:
        file_names = directory_file_names(path, recursive, exceptions)

    return file_names
    
def input_file_names(args):
    return file_names_at_path(args["input"], args["recursive"] == True)


###############################################################################
#
# Simple method to open a unicode CSV file.  If column names are provided the
# method also writes them out as the first line of the CSV
#
###############################################################################

def open_csv_file(name, column_names=None):
    output_file = open(name, "wb")
    output_csv_file = csv.writer(output_file,
                                 quoting=csv.QUOTE_MINIMAL)

    if column_names is not None:
        output_csv_file.writerow(column_names)

    return output_csv_file


def read_csv(name):
    rows = []
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            #remove the Latin spaces
            row[1] = row[1].replace(u'\xa0', u' ')
            rows.extend([row])

    return rows