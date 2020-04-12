import os
import re
import sys


def dict_add(d, file_path, function_name, index1, index2, function_line, hash, chunk):

    #Identify unique methods by file index, chunk line, and function declaration
    if index1 in d:
        if chunk in d[index1]:

            if function_line in d[index1][chunk]:
                d[index1][chunk][function_line]['fix_size'] += 1
            else:
                d[index1][chunk][function_line] = { 'index2':index2,'file_path':file_path,'function_name':function_name,'hash':hash,'fix_size': 1}
            return d

        else:
            d[index1][chunk] = dict()
    else:
        d[index1] = dict()

    return dict_add(d, file_path, function_name, index1, index2, function_line, hash, chunk)


def output_new_lines(d):
    for i in d:
        for c in d[i]:
            for f in d[i][c]:
                sys.stdout.write(i+","+d[i][c][f]['index2']+","+d[i][c][f]['function_name']+","+str(d[i][c][f]['fix_size'])+","+d[i][c][f]['file_path']+","+d[i][c][f]['hash']+","+f+","+c+"\n")

if __name__ == '__main__':

    try:

        chunk_re = "@@ .*[0-9]*,[0-9]*.*[0-9]*,[0-9]* @@"
        function_re = "[a-zA-Z0-9]*\(.*\)( throws [a-zA-Z]* )? ?{"
        function_re2 = "[a-zA-Z0-9]*\(.*\)( throws [a-zA-Z]* )? ?\n"
        #function_re = "(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$_\w<>\[\]\s]*\s+[\$_\w]+\([^\)]*\)?\s*\{?[^\}]"
        index_re = "index [0-9a-zA-Z]*\.\.[0-9a-zA-Z]*"

        modified_functions = dict()
        hash = sys.argv[1]

        function_line = ""
        function_name = ""
        bracket_count = 0
        file_path = ""
        indexes = ""
        chunk = ""

        function_re2_match = ""



        for line in sys.stdin:

            if line.startswith("+++"):
                #Only collect src/main files
                file_path = line.split("src")[-1].replace('\n','')
                function_name = ""
                bracket_count = 0

            index_search = re.search(index_re, line)
            if index_search:
                indexes = index_search.group(0).split(' ')[-1].split('..')

            chunk_search = re.search(chunk_re,line)
            if chunk_search:
                function_name = ""
                chunk = line.split("@@")[-1].replace(',','<comma>').replace('\n','')[1:]
                bracket_count = 0


            # Sometimes the open bracket of a function is on the second line,
            # Identifying methods of this type takes 2 steps
            if re.search("[ \t]*{\n",line) and function_re2_match != "":
                function_line = function_re2_match.group(0).replace(',','<comma>').replace('\n','')
                function_name = function_re2_match.group(0).split('(')[0].split(' ')[-1]
                bracket_count = 0
            #Try to match a function declaration without the {
            #If found and the next line is a bracket, the IF above will capture the function
            function_re2_match = ""
            f2_search = re.search(function_re2,line)
            if f2_search:
                function_re2_match = f2_search

            #Find function name, new functions dont count
            f_search = re.search(function_re,line)
            if f_search and not line.startswith('+'):
                function_line = f_search.group(0).replace(',','<comma>')
                function_name = f_search.group(0).split('(')[0].split(' ')[-1]
                bracket_count = 0

            if not line.startswith('+'):
                bracket_count += line.count('{') - line.count('}')

            #Find modified lines, but exclude new line additions or deletions
            if line.startswith('+ ') or line.startswith('- ') or line.startswith('+\t') or line.startswith('-\t'):
                if bracket_count > 0 and len(function_name) > 0:# and file_path.startswith("/main/") :
                    modified_functions = dict_add(modified_functions,file_path,function_name,indexes[0],indexes[1],function_line,hash,chunk)

        output_new_lines(modified_functions)

    except Exception as e:
        sys.exit()
