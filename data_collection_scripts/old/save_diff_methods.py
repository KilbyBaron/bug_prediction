import os
import re
import sys

#Test command
# git diff --no-prefix -U10000 c62529142b9ded096a1080d82c6934c3a309690c^! | python3 "../../c2v_models/scripts/save_diff_methods.py"


if __name__ == '__main__':

    try:

        #function_re = "[a-zA-Z0-9]*\(.*\)( throws [a-zA-Z]* )? ?{"
        function_re2 = "[a-zA-Z0-9]*\(.*\)( throws [a-zA-Z]* )? ?\n"
        function_re = "(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$_\w<>\[\]\s]*\s+[\$_\w]+\([^\)]*\)?\s*\{?[^\}]"

        bracket_count = 0
        function_re2_line = ""
        in_function=False
        function_contents = ""

        functions = []

        hash = sys.argv[1]

        for line in sys.stdin:

            # Sometimes the open bracket of a function is on the second line,
            # Identifying methods of this type takes 2 steps
            if re.search("[ \t]*{\n",line) and function_re2_line != "":
                bracket_count = 0
                in_function=True
                function_contents = function_re2_line

            #Try to match a function declaration without the {
            #If found and the next line is a bracket, the IF above will capture the function
            function_re2_line = ""
            if re.search(function_re2,line):
                function_re2_line = line

            #Find function name, new functions dont count
            if re.search(function_re,line) and not line.startswith('+'):
                bracket_count = 0
                in_function=True

            if not line.startswith('+'):
                bracket_count += line.count('{') - line.count('}')

            if in_function:
                if bracket_count > 0:
                    function_contents += line
                else:
                    if "\n+" in function_contents or "\n-" in function_contents:
                        functions.append("\n"+function_contents+line)
                    bracket_count = 0
                    in_function=False
                    function_contents = ""


        counter = 0

        for func in functions:

            counter += 1

            buggy = ""
            clean = ""

            lines = func.split('\n')
            fix_size = 0

            for line in lines:
                if line.startswith('+'):
                    clean = clean + line.replace('+','',1) + "\n"
                    fix_size += 1
                elif line.startswith('-'):
                    buggy = buggy + line.replace('-','',1) + "\n"
                    fix_size += 1

                else:
                    clean = clean + line + "\n"
                    buggy = buggy + line + "\n"


            with open("../../c2v_models/files/test/"+hash[:-2]+"_"+str(fix_size)+"_buggy_"+str(counter)+".java",'w') as new_buggy:
                new_buggy.write(buggy)
            with open("../../c2v_models/files/test/"+hash[:-2]+"_"+str(fix_size)+"_clean_"+str(counter)+".java",'w') as new_clean:
                new_clean.write(clean)




    except Exception as e:
        print(e)
        sys.exit()
