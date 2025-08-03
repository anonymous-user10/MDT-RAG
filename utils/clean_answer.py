
import re


def quoto_escape_in_json_str(text):
    try:
        ret = re.sub(r"(?<![\[\]\{\}\s\:\,])'s", r"\\'s", text)# 处理's
        ret_eval = eval(ret)
    except:
        try:
            ret = re.sub(r"(?<![\[\]\{\}\s\:\,])s'(?![\[\]\,\{\}\:])", "s\\'", ret) # 处理s'
            ret_eval = eval(ret)
        except:
            return ret
    return ret_eval
def remove_extra_brackets(s):
    num_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for num in num_list:
        pos = s.find(f'"tree{num}":')
        if pos != -1:
            while pos > 0 and s[pos] != '}':
                pos -= 1
            s = s[:pos] + s[pos+1:]
            try:
                s_temp = complete_brackets(s)
                result = eval(s_temp)
                return s_temp
            except:
                pass
        else:
            break
    return s

def complete_brackets(s):
    stack = []
    output = []
    for i, c in enumerate(s):
        if c in {'{', '['}:
            stack.append(c)
            output.append(c)
        elif c == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                output.append(c)
        elif c == ']':
            if stack and stack[-1] == '[':
                if i < len(s) - 1 and len(stack)==1:
                    pass
                else:
                    stack.pop()
                    output.append(c)
        else:
            output.append(c)  
    complement = []
    while stack:
        left = stack.pop()
        if left == '{':
            complement.append('}')
        else:
            complement.append(']')
    output += complement
    result =  ''.join(output)

    try :
        result_eval = eval(result)
        return result
    except:

        s = quoto_escape_in_json_str(result)
        s = str(s)
        return s



def clean_data(text,q_id=None)->dict:

    pos = True
    text = text.strip()
    def check_format(text):
        nonlocal pos
        try :
            tmp = eval(text)
            if isinstance(tmp, dict):
                return tmp
            else:
                if q_id: 
                    pos = False
                return {}
        except:
            if q_id:
                pos = False
            return {}
    
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3]
        result = check_format(text)
    elif text.startswith('{') and text.endswith('}'):
        result = check_format(text)
    else:
        ptn = re.compile(r'```json(.*?)```', re.S)
        ret = re.search(ptn, text)
        if ret:
            result = check_format(ret.group(1))
        else:
            # search json data in text start with { and end with }
            start_index = text.find('{')
            end_index = text.rfind('}')
            if start_index != -1 and end_index != -1:
                ret = text[start_index :end_index+1]
                result = check_format(ret)
            else:
                pos = False
                result = {}
    if result == {} and len(text) > 5:
        if text.count('{') <= 1:
            pass
        else:
            text_tmp = complete_brackets(text)
            result = check_format(text_tmp)
            if result != {}:
                pos = True
            else:
                text_tmp = remove_extra_brackets(text)
                result = check_format(text_tmp)
                if result != {}:
                    pos = True
                    
    if q_id and not pos:
        return [q_id,result]
    else:
        return result
    
if __name__ == '__main__':
    text = "{'stem':'tom's apples', 'price': 10, 'sdescription': 'best apples' '}"
    new_text = quoto_escape_in_json_str(text)
    print(new_text)
    
    