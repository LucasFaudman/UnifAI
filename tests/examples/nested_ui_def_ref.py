from unifai import UnifAIClient, tool, EvalSpec
from _provider_defaults import PROVIDER_DEFAULTS

import webbrowser
import tempfile
from pathlib import Path

return_ui_component = {
    "type": "function",
    "function": {
            "name": "return_ui_component",
            "description": "Return a UI component",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The type of the UI component",
                        "enum": ["div", "button", "header", "section", "field", "form", "input", "span", "script"]
                    },
                    "text_content": {
                        "type": "string",
                        "description": "The text content of the UI component. For script tags, this should be the script content."
                    },
                    "children": {
                        "type": "array",
                        "description": "Nested UI components",
                        "items": {
                            "$ref": "#"
                        }
                    },
                    "attributes": {
                        "type": "array",
                        "description": "Arbitrary attributes for the UI component, suitable for any element",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the attribute, for example onClick or className"
                                },
                                "value": {
                                    "type": "string",
                                    "description": "The value of the attribute"
                                }
                            },
                        "additionalProperties": False,
                        "required": ["name", "value"]
                        }
                    }
                },
                "required": ["type", "text_content", "children", "attributes"],
                "additionalProperties": False
            }
        }
}


get_ui_component = EvalSpec(
    eval_type="return_ui_component",
    system_prompt="Your role is to return a UI component based on the description provided.",
    tools=["return_ui_component"],
    tool_choice="return_ui_component",
    return_on="tool_call",
    return_as="last_tool_call_args",
)


def get_html_from_result(ui_component):
    html = ""
    html += f"<{ui_component['type']} "
    for attribute in ui_component['attributes']:
        html += f"{attribute['name']}='{attribute['value']}' "
    html += ">"
    if ui_component['text_content']:
        html += ui_component['text_content']
    if ui_component['children']:
        for child in ui_component['children']:
            html += get_html_from_result(child)
    html += f"</{ui_component['type']}>"
    return html


def render_html_in_browser(html_content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        temp_file.write(html_content.encode('utf-8'))
        temp_file.flush()        
        temp_file_path = Path(temp_file.name).resolve()
    
    # Open the file in the default web browser
    webbrowser.open(f'file://{temp_file_path}')
    input("Press Enter to delete...")
    temp_file_path.unlink()

if __name__ == "__main__":
    ai = UnifAIClient(
    provider_client_kwargs={
        "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1]
    },
    tools=[return_ui_component],
    eval_prameters=[get_ui_component]
    )

    # default_content = """Create a login form with a username and password entries, and a submit button. 
    # Space the elements out with a margin of 10px.
    # Make the username and password fields required and the password field should be of type 'password'.
    # Make the colors look professional and easy on the eyes.
    # Add some JavaScript to create an alert and show a cool animation when the submit button is clicked.
    # Add a header with the text 'Login Form' above the form.
    # """

    # default_content = "Use Javascript and HTML to create a fully playable Tic-Tac-Toe game with a 3x3 grid and a reset button. The game should be interactive, responsive and include an animation when a player wins. The game should have a header with the text 'Tic-Tac-Toe' above the grid."
    default_content = "Use Javascript and HTML to create a fully playable Connect 4 game with a 7x6 grid and a reset button. The game should be interactive, responsive and include an animation when a player wins. The game should have a header with the text 'Connect 4' above the grid."
    default_content = "Use Javascript and HTML to implement a fully playable First Person Shooter game like DOOM. The game should be playable, interactive and responsive. The game should be visible inside a div with id='game-container'. The game should be interactive, responsive and include an animation when a player wins. The game should have a start button that starts the game and reset button that resets the game state. The game should show the player's health, ammo counta and number of enemies killed. Enemies should spawn randomly and move towards the player. The player should be able to shoot the enemies and the enemies should disappear when shot. The player should be able to move around the game area using the arrow keys and shoot with the spacebar. IMPORTANT: The game should be fully playable and interactive first person shooter." 


    content = input("Describe a UI element: ") or default_content
    print("Creating...\n", content)
    result = ai.evaluate("return_ui_component", content=content, provider="anthropic")
    print("Result: ", result)
    html = get_html_from_result(result)
    print("HTML: ", html)
    print("Opening in browser...")
    render_html_in_browser(html)
