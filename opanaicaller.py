from openai import OpenAI, OpenAIError
from json import loads as json_loads, dumps as json_dumps, JSONDecodeError
from os import environ
from typing import Optional, Literal, Type, Callable, get_origin, get_args, List, Dict
from pydantic import BaseModel, Field
from collections import defaultdict


TOOLS = {
    "return_flagged_and_reason": {
        "type": "function",
                "function": {
                    "name": "return_flagged_and_reason",
                    "description": "Return a boolean indicating whether the content should be flagged and a concise reason for the flag if True.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flagged": {
                                "type": "boolean",
                                "description": "True if the content should be flagged, False otherwise."
                            },
                            "reason": {
                                "type": "string",
                                "description": "A concise reason for the flag if True. An empty string if False."
                            }
                        },
                        "required": ["flagged", "reason"]
                    }
                }
    },
    "return_screenshot_explanation": {
        "type": "function",
                "function": {
                    "name": "return_screenshot_explanation",
                    "description": "Return an explanation of why a screenshot was flagged as malicious after reviewing it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "explanation": {
                                "type": "string",
                                "description": "An explanation of why the screenshot was flagged as malicious. "
                                "Explain the tactics used by the malicious actors and the potential impact on users. "
                                "Explain the malicious elements and languange in the screenshot and how they indicate malicious activity."
                            },
                            "confidence_level": {
                                "type": "string",
                                "description": "High, Medium, or Low confidence in the flag and explanation being accurate. "
                            }
                        },
                        "required": ["explaination", "confidence_level"]
                    }
                }
    },    
}







PROMPT_REGISTRY = {
    "urlEval": {
        "system_prompt": (
        "You review HTML text to flag elements that may contain spam, misinformation, or other malicious items. "
        "You also check the associated URLS for signs of typosquatting or spoofing. "
        "Use the return_flagged_and_reason function to return your result."
        ),
        "tools": [TOOLS["return_flagged_and_reason"]],
        "tool_choice": "return_flagged_and_reason",
        "few_shot_prompting_examples": []
    },
        
    "emailEval": {
        "system_prompt": (
        "You review components from email lines in an inbox and flag ones that may contain spam, phishing, or other malicious items. "
        "Use the return_flagged_and_reason function to return your result."
        ),
        "tools": [TOOLS["return_flagged_and_reason"]],
        "tool_choice": "return_flagged_and_reason",
        "few_shot_prompting_examples": []
    },
    "screenshotEval": {
        "system_prompt": (
        "Your role is to review screenshots of websites to flag elements that may contain spam, misinformation, or other malicious items. "
        "Use the return_flagged_and_reason function to return your result."
        ),
        "tools": [TOOLS["return_flagged_and_reason"]],
        "tool_choice": "return_flagged_and_reason",
        "few_shot_prompting_examples": []
    },   
    "screenshotExplain": {
        "system_prompt": (
        "Your role is to review screenshots of websites and explain why it contains spam, misinformation, or other malicious items. "
        "Use the return_screenshot_explanation function to return your result."
        ),
        "tools": [TOOLS["return_screenshot_explanation"]],
        "tool_choice": "return_screenshot_explanation",
        "few_shot_prompting_examples": []
    },        
}


class OpenAIHandler:

    def __init__(self, 
                openai_client_kwargs: Optional[dict] = None, 
                model: Optional[str] = None,
                prompt_registry: Optional[dict] = None,
                default_completion_kwargs: Optional[dict] = None
                ):
        self.client = OpenAI(**(openai_client_kwargs or {}))
        self.model = model or environ.get('OPENAI_MODEL', 'gpt-4o')
        self.prompt_registry = prompt_registry or PROMPT_REGISTRY
        self.default_completion_kwargs = default_completion_kwargs or {
            "timeout": 30,
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

    def recursively_make_serializeable(self, obj):
        """Recursively makes an object serializeable by converting it to a dict or list of dicts and converting all non-string values to strings."""
        serializeable_types = (str, int, float, bool, type(None))
        if isinstance(obj, dict):
            return {k: self.recursively_make_serializeable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursively_make_serializeable(item) for item in obj]
        elif not isinstance(obj, serializeable_types):
            return str(obj)
        else:
            return obj

    def format_content(self, content):
        """Formats content for use a message content. If content is not a string, it is converted to json."""
        if not isinstance(content, str):
            content = self.recursively_make_serializeable(content)
            content = json_dumps(content, indent=0)

        return content

    def make_few_shot_prompt(self, system_prompt, examples, user_input):
        """Makes list of message objects from system prompt, examples, and user input."""

        # Join system_prompt if it is a list or tuple
        if not isinstance(system_prompt, str):
            system_prompt = " ".join(system_prompt)

        if not isinstance(user_input, str):
            # JSON encode user_input if it is not a string
            user_input = self.format_content(user_input)

        # Create list of messages beginning with system_prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add examples to messages
        for example in examples:
            example_input = self.format_content(example['input'])
            example_response = self.format_content(example['response'])

            messages.append({"role": "user", "content": example_input})
            messages.append({"role": "assistant", "content": example_response})

        # Add user_input to messages
        messages.append({"role": "user", "content": user_input})

        return messages

    def get_response(self, ai_func, content) -> dict:
        """Gets a response from the OpenAI API for a given analysis type and content."""
        
        prompt_data = self.prompt_registry.get(ai_func)
        if not prompt_data:
            return {"error": f"Analysis type '{ai_func}' is not supported."}
        
        messages = self.make_few_shot_prompt(
            system_prompt=prompt_data['system_prompt'], 
            examples=prompt_data['few_shot_prompting_examples'],
            user_input=content,
        )

        completion_kwargs = {
            **self.default_completion_kwargs,
            "model": self.model,
            "messages": messages,
            "n": 1,
        }
        if (tools := prompt_data['tools']):
            completion_kwargs['tools'] = tools
        if (tool_choice := prompt_data['tool_choice']):
            if tool_choice in ("auto", "none"):
                completion_kwargs['tool_choice'] = tool_choice
            else:
                completion_kwargs['tool_choice'] = {"type": "function", "function": {"name": tool_choice}}   
                
        try:
            completion_response = self.client.chat.completions.create(**completion_kwargs)
            return json_loads(completion_response.choices[0].message.tool_calls[0].function.arguments)
            # response_data = {
            #     "result": json_loads(completion_response.choices[0].message.tool_calls[0].function.arguments),
            #     "usage": {
            #         "model": completion_response.model,
            #         "usage": completion_response.usage.model_dump(),                 
            #     }
            # }
            # return response_data
        except (OpenAIError, IndexError, JSONDecodeError) as e:
            return {"error": str(e)}
        


    def get_func_response(self, ai_func, content) -> dict:
        """Gets a response from the OpenAI API for a given analysis type and content."""
        
        prompt_data = self.prompt_registry.get(ai_func)
        if not prompt_data:
            return {"error": f"Analysis type '{ai_func}' is not supported."}
        
        messages = self.make_few_shot_prompt(
            system_prompt=prompt_data['system_prompt'], 
            examples=prompt_data['few_shot_prompting_examples'],
            user_input=content,
        )

        completion_kwargs = {
            **self.default_completion_kwargs,
            "model": self.model,
            "messages": messages,
            "n": 1,
        }
        if (tools := prompt_data['tools']):
            completion_kwargs['tools'] = tools
        if (tool_choice := prompt_data['tool_choice']):
            if tool_choice in ("auto", "none"):
                completion_kwargs['tool_choice'] = tool_choice
            else:
                completion_kwargs['tool_choice'] = {"type": "function", "function": {"name": tool_choice}}   
                
        try:
            completion_response = self.client.chat.completions.create(**completion_kwargs)
            json_loads(completion_response.choices[0].message.tool_calls[0].function.arguments)
            # response_data = {
            #     "result": json_loads(completion_response.choices[0].message.tool_calls[0].function.arguments),
            #     "usage": {
            #         "model": completion_response.model,
            #         "usage": completion_response.usage.model_dump(),                 
            #     }
            # }
            # return response_data
        except (OpenAIError, IndexError, JSONDecodeError) as e:
            return {"error": str(e)}
        

AIType = Literal["object", "array", "string", "number", "boolean", "null"]

class AIParam(BaseModel):    
    name: str
    description: str
    type: AIType = "string"
    required: bool = False
    options: list[str] = Field(default_factory=list)

    def to_dict_item(self):
        return self.name, { 
            "type": self.type, 
            "description": self.description 
        }
    
class AIObjectParam(AIParam):
    type: Literal["object"] = "object"
    properties: dict[str, AIParam]
    
    def to_dict_item(self):
        return self.name, { 
            "properties": {k: v.to_dict_item() for k, v in self.properties.items()} ,
            "required": [k for k, v in self.properties.items() if v.required]
        }
    
class AIArrayParam(AIParam):
    type: Literal["array"] = "array"
    items: AIParam
    
    def to_dict_item(self):
        return self.name, { 
            "items": self.items.to_dict_item() 

        }
    
def parse_ai_func_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """Parses a docstring for an AI function and returns a dictionary of parameter descriptions."""

    prompt = docstring.split("<PROMPT>", 1)[-1].split("</PROMPT>", 1)[0].strip()
    func_description, params_text = map(str.strip, prompt.split("<PARAMS>", 1))
    param_defs = defaultdict(str)
    for line in params_text.split("\n"):
        if not (line := line.strip()) or line.count(": ") < 1:
            continue

        name, description = line.split(": ", 1)
        param_defs[name] = description

    print(func_description)
    print(param_defs)
    return func_description, param_defs


PY2AI_TYPES: dict[Type, AIType] = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

def translate_annotation(annotation: Type) -> tuple[AIType, AIType|None, bool]:
    required = True
    if annotation.__name__ == "Optional":
        required = False
        annotation = get_args(annotation)[0]

    ai_type = PY2AI_TYPES.get(annotation, "string")

    # ai_type = PY2AI_TYPES.get(annotation)
    # if not ai_type:
    #     raise ValueError(f"Type '{annotation.__name__}' is not supported.")

    if ai_type == "array":
        args = get_args(annotation)
        if len(args) != 1:
            raise ValueError("Array types must have one type argument.")
        
        item_type = args[0]
        return ai_type, translate_annotation(item_type)[0], required

    if ai_type == "object":
        args = get_args(annotation)
        if len(args) != 2:
            raise ValueError("Dict/Object types must have two type arguments.")

        key_type, value_type = args
        if key_type != str:
            raise ValueError("Dict/Object keys must be strings.")
        return ai_type, translate_annotation(value_type)[0], required
    
    return ai_type, None, required

    




class AIFunction:

    def __init__(self, 
                 func: Callable,
                 description: Optional[str] = None,
                 parameter_defs: Optional[dict[str, str|dict]] = None,  
                 examples: Optional[list[dict[str, str]]] = None,
                 ):
        self.func = func
        self.name = func.__name__
        self.description = description
        parameter_defs = parameter_defs or {}

        # Parse docstring for function description and parameter descriptions        
        if (not description or not parameter_defs) and func.__doc__:
            doc_description, doc_parameters = parse_ai_func_docstring(func.__doc__)
            self.description = description or doc_description
            parameter_defs.update(doc_parameters)

        if not parameter_defs:
            raise ValueError("Function must have at least one parameter.")
        
        self.parameters = {}
        annotations = func.__annotations__

        for param_name, param_type in annotations.items():
            if param_name == "return":
                continue

            if not (param_description := parameter_defs.get(param_name)):
                raise ValueError(f"Parameter '{param_name}' does not have a description.")
            
            ai_type, item_type, required = translate_annotation(param_type)
            
            
            if ai_type == "array" and item_type:
                array_description = param_description.get("description", "")
                item_description = param_description.get("item_description", "")
                param = AIArrayParam(name=param_name, description=array_description, required=required, items=AIParam(name=param_name, description=item_description, type=item_type))

            elif ai_type == "object" and item_type:
                object_description = param_description.get("description", "")
                properties = {}
                for k, v in item_type.__annotations__.items():
                    if k == "return":
                        continue
                    if not (item_description := param_description.get(k)):
                        raise ValueError(f"Parameter '{k}' does not have a description.")
                    ai_type, item_type, required = translate_annotation(v)
                    properties[k] = AIParam(name=k, description=item_description, type=ai_type, required=required)
                param = AIObjectParam(name=param_name, description=object_description, required=required, properties=properties)

            elif isinstance(param_description, dict):
                param_description = param_description.get("description", "")
                param = AIParam(name=param_name, description=param_description, type=ai_type, required=required)
            else:
                param = AIParam(name=param_name, description=param_description, type=ai_type, required=required)

            self.parameters[param_name] = param


    def to_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {k: v.to_dict_item() for k, v in self.parameters.items()}
            }
        }
    
    def __get__(self, instance: OpenAIHandler, content, **kwargs):
        response = instance.get_func_response(self, content, **kwargs)
        return self.func(**response)
    

def ai_function(  
        description: Optional[str] = None,
        param_defs: Optional[dict[str, str|tuple|dict]] = None,
        examples: Optional[list[dict[str, str]]] = None,        
        ):
    def decorator(func: Callable):
        return AIFunction(func, description, param_defs, examples)
    return decorator



class SecurityAI(OpenAIHandler):

    @ai_function(
        description="Return a boolean indicating whether the content should be flagged and a concise reason for the flag if True.",
        param_defs={
            "flagged": "True if the content should be flagged, False otherwise.",
            "reason": "A concise reason for the flag if True. An empty string if False.",
            "source_list": {
                "description": "A list of sources to check for the content.", 
                "item_description": "A URL or other source to check for the content."
            }
        }
    )
    def return_flagged_and_reason(self, 
                                  flagged: Optional[bool], 
                                  reason: str,
                                  source_list: List[str],
                                  ):
        """Return a boolean indicating whether the content should be flagged and a concise reason for the flag if True."""        
        
        print("Flagged:", flagged)
        print("Reason:", reason)
        print("Sources:", source_list)
        return flagged, reason, source_list
    

    # @ai_function
    # def return_flagged_and_reason(self, 
    #                               flagged: Optional[bool], 
    #                               reason: str):
    #     """
    #     <PROMPT>

    #     Return a boolean indicating whether the content should be flagged and a concise reason for the flag if True.
        
    #     <PARAMS>
    #     flagged: True if the content should be flagged, False otherwise.
    #     reason: A concise reason for the flag if True. An empty string if False.
    #     </PARAMS>
        
    #     </PROMPT>
    #     """        
        
    #     print("Flagged:", flagged)
    #     print("Reason:", reason)
    #     return flagged, reason    
    




# print(parse_ai_func_docstring(ds1))
# print(parse_ai_func_docstring(ds2))
# print(parse_ai_func_docstring(ds3))



if __name__ == "__main__":
    ai = SecurityAI()
    print(ai.return_flagged_and_reason("gooogle.com"))