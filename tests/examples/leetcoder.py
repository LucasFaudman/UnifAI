
from typing import Any
from unifai import UnifAIClient, tool, Message, EvalSpec, Tool, NumberToolParameter, StringToolParameter, ArrayToolParameter, ObjectToolParameter
from souperscraper import SouperScraper, Keys
from pydantic import BaseModel

from _provider_defaults import PROVIDER_DEFAULTS

# class LeetCodeProblem(BaseModel):
#     number: int
#     title: str
#     difficulty: str
#     description: str
#     constraints: str
#     examples: list[str]
#     test_cases: list[dict[str, str]]

# extract_leetcode_problem = Tool(
#     name="extract_leetcode_problem",
#     description="Return a leetcode problem from unstuctured text.",
#     parameters={
#         "number" : NumberToolParameter(description="The problem number."),
#         "title" : StringToolParameter(description="The problem title."),
#         "difficulty" : StringToolParameter(description="The problem difficulty.", enum=["easy", "medium", "hard"]),
#         "description" : StringToolParameter(description="The problem description."),
#         "constraints" : StringToolParameter(description="The problem constraints."),
#         "examples" : ArrayToolParameter(items=StringToolParameter(description="The problem examples.")),
#         "test_case_inputs" : ArrayToolParameter(
#             items=ObjectToolParameter(
#                 name="keyword_type_pair",
#                 properties=[
#                     StringToolParameter(name="keyword", description="The keyword."),
#                     StringToolParameter(name="type", description="The type.", enum=["int", "float", "str", "list", "dict"])
#                 ]
#             )),            
#     }
# )

extract_leetcode_problem = EvalSpec(
    eval_type="extract_leetcode_problem",
    system_prompt="Your role is to extract leetcode problems from the leetcode website.",
    tools=[
        Tool(
            name="extract_leetcode_problem",
            description="Extract a leetcode problem from unstuctured text.",
            parameters={
                "number" : NumberToolParameter(description="The problem number."),
                "title" : StringToolParameter(description="The problem title."),
                "difficulty" : StringToolParameter(description="The problem difficulty.", enum=["easy", "medium", "hard"]),
                "description" : StringToolParameter(description="The problem description."),
                "constraints" : StringToolParameter(description="The problem constraints."),
                "examples" : ArrayToolParameter(items=StringToolParameter(description="The problem examples.")),
                "test_case_inputs" : ArrayToolParameter(
                    items=ObjectToolParameter(
                        name="keyword_type_pair",
                        properties=[
                            StringToolParameter(name="keyword", description="The keyword."),
                            StringToolParameter(name="type", description="The type.", enum=["int", "float", "str", "list", "dict"])
                        ]
                    )),            
            }
        )
    ],
    tool_choice="extract_leetcode_problem",
    return_as="last_tool_call_args"
)

# solve_leetcode_problem = EvaluateParameters(
#     eval_type="solve_leetcode_problem",
#     system_prompt="Your role is to solve leetcode problems using the optimal solution. You can add/modify test cases and submit your solution.",
#     tools=[

#     ],
#     tool_choice="required",
#     return_as="last_tool_call_args"
# )

# extract_and_summarize_solution = Tool(
#     name="extract_and_summarize_solution",
#     description="Extract and summarize the final solution including its runtime and memory usage.",
#     parameters={
#         "final_solution" : StringToolParameter(description="The final solution."),
#         "runtime_ms" : NumberToolParameter(description="The runtime in milliseconds."),
#         "runtime_percentile" : NumberToolParameter(description="The runtime percentile."),
#         "memory_mb" : NumberToolParameter(description="The memory in megabytes."),
#         "memory_percentile" : NumberToolParameter(description="The memory percentile."),
#     }
# )


class LeetScraper:
    WEBDRIVER_PATH = "/Users/lucasfaudman/.chromedriver/chromedriver-mac-x64/chromedriver"

    def __init__(self, 
                 ai: UnifAIClient,
                 chromedriver_path: str = WEBDRIVER_PATH
                 ):
        self.ai = ai
        self.scraper = SouperScraper(
            executable_path=chromedriver_path, 
            # user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            selenium_options_args=[
                "--disable-blink-features=AutomationControlled",
                "--exclude-switches=enable-automation",
                "--use-automation-extension=false",
                "--profile-directory=Lucas",
                ]
        
        )

    def login(self):
        self.scraper.goto("https://www.google.com/search?q=leetcode login")
        input("Press enter to continue after logging in.")
        self.scraper.goto("https://leetcode.com/problemset/")

    def find_problem_hrefs(self):
        problem_href = lambda href: href and href.startswith("/problems/")
        problems = [href["href"] for href in self.scraper.soup.find_all("a", href=problem_href)]
        return problems


    def goto_problem(self, problem_href: str):
        self.scraper.goto(f"https://leetcode.com{problem_href}", sleep_secs=2)
        
    def select_language(self, language: str):
        input("Press enter to continue after selecting language.")
        # self.scraper.wait_for_visibility_of_element_located_by_id("headlessui-popover-button-:r1g:").click()
        # self.scraper.find_element_by_text(language).click()

    def extract_problem(self):
        problem_dict = self.ai.evaluate(
            eval_type="extract_leetcode_problem",
            content=self.scraper.soup.text
        )
        return problem_dict

    def get_current_solution(self):
        return self.scraper.find_element_by_class_name("editor-scrollable").text

    def show_testcase_container(self):
        if not self.scraper.wait_for_visibility_of_element_located_by_css_selector("div[data-layout-path='/c1/ts1/t0']"):
            test_container = self.scraper.find_element_by_id('testcase_tabbar_outer')
            test_container.click()
            test_container.find_elements("css selector", "button")[1].click()

    def get_test_cases(self):
        # test_container = self.scraper.find_element_by_id('testcase_tabbar_outer')
        # test_container.click()
        # test_container.find_elements("css selector", "button")[1].click()
        self.show_testcase_container()
        test_case_container = self.scraper.find_element_by_css_selector("div[data-layout-path='/c1/ts1/t0']")
        test_case_container_text = test_case_container.text
        case_names = [name for name in test_case_container.text.split("\n") if name.startswith("Case")]

        test_cases = {}
        for case_name in case_names:
            self.scraper.find_element_by_text(case_name).click()
            test_case_container = self.scraper.find_element_by_css_selector("div[data-layout-path='/c1/ts1/t0']")
            test_case_container_text = test_case_container.text
            case_lines = test_case_container_text.replace(" =\n", "=").split("\n")[len(case_names):-1]            
            test_cases[case_name] = dict(line.split("=") for line in case_lines)

        return test_cases


    def add_test_case(self, test_case: dict):
        # test_container = self.scraper.find_element_by_id('testcase_tabbar_outer')
        # test_container.click()
        # test_container.find_elements("css selector", "button")[1].click()
        self.show_testcase_container()
        test_case_container = self.scraper.find_element_by_css_selector("div[data-layout-path='/c1/ts1/t0']")
        test_case_container.find_element("css selector", "button[data-state='closed']").click()
        case_num = sum(1 for name in test_case_container.text.split("\n") if name.startswith("Case")) + 1
        self.update_test_case(str(case_num), test_case)


    def update_test_case(self, case_number: str, test_case: dict):
        self.scraper.find_element_by_text(f"Case {case_number}").click()
        test_case_container = self.scraper.find_element_by_css_selector("div[data-layout-path='/c1/ts1/t0']")
        labels = test_case_container.find_elements("class name", "text-xs")[:-2]
        inputs = test_case_container.find_elements("css selector", "div[contenteditable='true']")
        for label, input in zip(labels, inputs):
            label_text = label.text.strip(' =')
            input.send_keys(Keys.BACKSPACE * len(input.text))
            input.send_keys(test_case[label_text])


    def update_solution(self, solution: str):
        """Update the solution in the editor.

        Args:
            solution (str): Updated solution to the problem that will replace the current solution. Be sure to format and indent the solution properly.
        """
        input_area = self.scraper.find_element_by_class_name("inputarea")
        self.scraper.scroll_to(input_area)
        input_area.click()
        for _ in self.get_current_solution():
            input_area.send_keys(Keys.BACKSPACE)
            input_area.send_keys(Keys.BACK_SPACE)
            input_area.send_keys(Keys.DELETE)                    
        input_area.send_keys(200 * Keys.BACKSPACE)
        input_area.send_keys(200 * Keys.DELETE)

        # for line in solution.splitlines():
        #     input_area.send_keys(line)
        #     input_area.send_keys(Keys.RETURN)
        # input_area.send_keys(solution)
        for char in solution:
            input_area.send_keys(char + Keys.DELETE)
        input_area.send_keys(Keys.RETURN)
        input_area.send_keys(200 * Keys.DELETE)

        return self.run_tests()   
            

    def wait_for_judging(self):
        self.scraper.wait_for_visibility_of_element_located_by_css_selector("img[alt='Pending...']")
        self.scraper.wait_for_invisibility_of_element_located_by_css_selector("img[alt='Pending...']")
        self.scraper.wait_for_visibility_of_element_located_by_css_selector("img[alt='Judging...']")
        self.scraper.wait_for_invisibility_of_element_located_by_css_selector("img[alt='Judging...']")


    def run_tests(self):
        self.scraper.find_element_by_text("Run").click()
        self.wait_for_judging()
        test_results_container = self.scraper.find_element_by_css_selector("div[data-layout-path='/c1/ts1/t1']")
        result = test_results_container.text
        self.scraper.find_element_by_css_selector("div[data-layout-path='/c1/ts1/tb0']").click()
        return result


    def submit_solution(self):
        self.scraper.find_element_by_text("Submit").click()
        self.wait_for_judging()
        submit_results_container = self.scraper.find_element_by_css_selector("div[data-layout-path='/ts0/t1']").find_element("class name", "space-y-4")
        submit_results_container.text.replace("\nEditorial", "").replace("\nUse Testcase", "").replace("\nThis doesn't support visualization.", "")
        return submit_results_container.text


    def add_test_case_tool_from_test_cases(self, test_case: dict):
        return Tool(
            name="add_test_case",
            description="Add a test case to the problem.",
            parameters=[
                ObjectToolParameter(
                    name="test_case",
                    properties=[
                        StringToolParameter(
                            name=key, 
                            description=f"Test value for {key}."
                        ) for key in test_case.keys()
                    ]
                )
            ],
            callable=self.add_test_case
        )
    
    def update_test_case_tool_from_test_cases(self, test_case: dict):
        return Tool(
            name="update_test_case",
            description="Update a test case",
            parameters={
                "case_number": NumberToolParameter(description="The case number. (1-7)"),
                "test_case": ObjectToolParameter(
                    name="test_case",
                    properties=[
                        StringToolParameter(
                            name=key, 
                            description=f"Testcase value for {key}."
                        ) for key in test_case.keys()
                    ]
                )
            },
            callable=self.update_test_case
        )    

    def make_prompt(self, 
                    problem: dict, 
                    test_cases: dict,
                    current_solution: str
                    ):
        prompt = f"Problem: {problem['title']} ({problem['difficulty']})\n\n"
        prompt += f"Description: {problem['description']}\n\n"
        prompt += f"Constraints: {problem['constraints']}\n\n"
        prompt += f"Examples: " + "\n".join(problem['examples']) + "\n\n"
        # prompt += f"Test Cases: " + "\n".join([f"{case['name']}: {case['input']} -> {case['output']}" for case in test_cases]) + "\n\n"
        prompt += f"Test Cases: " + "\n".join([f"{case_name}: {case}" for case_name, case in test_cases.items()]) + "\n\n"
        prompt += f"Current Solution: {current_solution}\n\n"
        print(prompt)
        return prompt


    def solve_problem(self, 
                      problem_href: str, 
                      language: str = "Python3",
                      min_runtime_percentile: float = 0.5,
                      min_memory_percentile: float = 0.5
                      ):
        
        self.goto_problem(problem_href)
        self.select_language(language)
        problem = self.extract_problem()
        solution = self.get_current_solution()
        test_cases = self.get_test_cases()


        solve_leetcode_problem = EvalSpec(
            eval_type="solve_leetcode_problem",
            system_prompt="Your role is to solve leetcode problems using the optimal solution. You can add/modify test cases and submit your solution.",
            tools=[
                self.update_solution,
                # self.add_test_case_tool_from_test_cases(test_cases),
                self.update_test_case_tool_from_test_cases(test_cases),
                self.run_tests,
                self.submit_solution,
                Tool(
                    name="extract_and_summarize_solution",
                    description="Extract and summarize the final solution including its runtime and memory usage.",
                    parameters={
                        "final_solution" : StringToolParameter(description="The final solution."),
                        "runtime_ms" : NumberToolParameter(description="The runtime in milliseconds."),
                        "runtime_percentile" : NumberToolParameter(description="The runtime percentile."),
                        "memory_mb" : NumberToolParameter(description="The memory in megabytes."),
                        "memory_percentile" : NumberToolParameter(description="The memory percentile."),
                    }
                )
            ],
            tool_choice="required",
            return_as="last_tool_call_args",
            return_on="extract_and_summarize_solution"
        )

        runtime_percentile, memory_percentile = 0, 0
        while runtime_percentile < min_runtime_percentile or memory_percentile < min_memory_percentile:
            prompt = self.make_prompt(problem, test_cases, solution)
            solution = self.ai.evaluate(
                eval_type=solve_leetcode_problem,
                content=prompt
            )
            if not solution:
                input("Failed to solve problem. Press enter to continue.")
                continue

            runtime_percentile = solution["runtime_percentile"]
            memory_percentile = solution["memory_percentile"]
            solution = self.get_current_solution()
            test_cases = self.get_test_cases()            
            print(f"Solved {problem['title']} with runtime percentile: {runtime_percentile} and memory percentile: {memory_percentile}")
            

        # self.add_test_case({"arr": "[1,2,3,4]", "k":"5"})
        # self.update_test_case("1", {"arr": "[1,2,3,4]", "k":"5"})
        # self.run_tests()
        # self.submit_solution()
        # self.update_solution(solution)
        # self.add_test_case(test_cases)
        # self.run_tests()
        # self.submit_solution()


if __name__ == "__main__":
    ai = UnifAIClient(
        provider_client_kwargs={
            "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
            # "google": PROVIDER_DEFAULTS["google"][1],
            "openai": PROVIDER_DEFAULTS["openai"][1],
            "ollama": PROVIDER_DEFAULTS["ollama"][1]
        },
        eval_prameters=[extract_leetcode_problem]
    )
    
    leet_scraper = LeetScraper(ai=ai)
    leet_scraper.login()
    problem_hrefs = leet_scraper.find_problem_hrefs()
    leet_scraper.solve_problem(problem_hrefs[0])
