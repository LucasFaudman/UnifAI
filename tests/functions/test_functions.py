import pytest
from unifai import UnifAI, FunctionConfig, BaseModel
from unifai.components.output_parsers.pydantic_output_parser import PydanticParser
from unifai.types import Message, Tool, ArrayToolParameter, ObjectToolParameter, BooleanToolParameter, StringToolParameter, NumberToolParameter
from basetest import base_test_llms_all, PROVIDER_DEFAULTS
from unifai.types.annotations import ProviderName
from typing import Literal
import httpx


@pytest.mark.parametrize("url, link_text, flagged", [
    ("https://google.com", "Google", False),
    ("https://g00gle.com", "Google", True),
    # ("https://github.com", "GitHub", False),
    # ("https://githu8.com", "GitHub", True),
    # ("https://microsoft.com", "Microsoft", False),
    # ("https://micros0ft.com", "Microsoft", True),
    # ("https://apple.com", "Apple", False),
    # ("https://app1e.com", "Apple", True),    
    # ("chromeupdater.com", "Chrome Updater", True),
])
@base_test_llms_all
def test_evalutate_flagged_reason(
    provider: ProviderName, 
    init_kwargs: dict, 
    func_kwargs: dict,
    url, 
    link_text,
    flagged: bool
    ):

    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])

    class FlaggedReason(BaseModel):
        flagged: bool
        """True if the content should be flagged, False otherwise."""
        reason: str
        """A concise reason for the flag if True. An empty string if False."""

        def print_reason(self):
            print(f"Flagged: {self.flagged}\nReason: {self.reason}")

    url_eval_config = FunctionConfig(
        name="urlEval",
        system_prompt="You review URLs and HTML text to flag elements that may contain spam, misinformation, or other malicious items. Check the associated URLS for signs of typosquatting or spoofing. ",
        prompt_template="URL:{url}\nLINK TEXT:{link_text}",
        output_parser=FlaggedReason,
    )

    url_eval = ai.get_function(url_eval_config)
    flagged_reason = url_eval(url=url, link_text=link_text)
    assert flagged_reason.flagged == flagged
    assert isinstance(flagged_reason.reason, str)
    assert isinstance(flagged_reason, FlaggedReason)
    print(f"{flagged_reason.flagged=} {flagged_reason.reason=}")
    flagged_reason.print_reason()



contacts_input_1 = """
In a landmark presentation at GlobalTech Innovations, Mr. Robert Hayes (robert.hayes@globaltech.com, +1-555-284-3489, 123 Innovation Drive, Silicon Valley, CA, USA), Senior Solutions Architect, outlined the future roadmap for AI-driven analytics. Robert emphasized the domestic rollout, noting a 95.2% confidence in initial predictions. His male perspective brought valuable insight to the team.

Meanwhile, Ms. Jennifer Li (j.li@finexus.co.uk, +44-20-7946-0011, 15 Kensington Road, London, UK), a Lead Data Analyst at Finexus and a female voice in the conversation, shared international growth metrics, with 80.5% confidence in global expansion. Jennifer's work has been pivotal in transitioning to cross-border financial solutions, which are predominantly international in scope.

At NextGen Synergy, Mr. Ethan Parker (e.parker@nextgen.com, +1-718-555-9922, 45 East Ave, Brooklyn, NY, USA), Chief Technical Officer, highlighted the rising need for scalable quantum networks. Ethan, a domestic male with 90.0% confidence, proposed a new smart logistics algorithm to streamline NextGen's operations.

Mrs. Emily Cheng (emily.cheng@logixconnect.org, +1-323-555-4545, 502 Industrial Lane, Los Angeles, CA, USA), a female Project Manager at LogixConnect, stressed the importance of cybersecurity compliance across multi-cloud infrastructures. Her analysis showed a 85.4% confidence in achieving full-scale domestic deployment within 18 months.

During a parallel discussion at Future Horizons, Ms. Olivia Ramirez (o.ramirez@futurehorizons.io, +61-3-5559-1276, 76 Harborview Street, Melbourne, VIC, Australia), female Head of Operations, emphasized the adoption of AI to enhance operational efficiency in urban planning. As an international expert, Olivia expressed 75.6% confidence in the initial deployment phase.

Across the boardroom, Dr. Michael Foster (michael.foster@innovatix.ai, +1-650-555-7783, 789 Quantum Park, Palo Alto, CA, USA), Senior Researcher at Innovatix, presented the latest breakthroughs in machine learning. Michael, a male with 93.1% confidence, noted that his domestic team's achievements were a result of unified research efforts.

Mr. Daniel Clark (d.clark@greenpower.solutions, +1-646-555-2837, 89 West 23rd St, New York, NY, USA), Chief Energy Officer at GreenPower Solutions, a domestic male, expressed 88.9% confidence in the rollout of their AI-powered renewable energy platform.

In a separate conversation, Ms. Sophia Evans (sophia.evans@cyberdynetech.co, +44-113-555-7612, 52 Leeds Ave, Leeds, UK), Senior AI Engineer at CyberdyneTech, underscored the importance of international collaboration to enhance machine learning protocols. Sophia, a female with 81.3% confidence, reflected on her team's work integrating domestic and international standards.

Finally, Mr. Liam Turner (liam.turner@stellarconnect.org, +61-2-5556-9876, 32 Circular Quay, Sydney, NSW, Australia), Director of Innovation at StellarConnect, addressed the need for AI-driven solutions in maritime logistics. Liam, an international male, delivered with 89.4% confidence that StellarConnect's new platform would revolutionize shipping logistics globally.

"""

contacts_input_2 = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corporate News</title>
</head>
<body>
    <h1>GlobalTech 2024 Innovation Conference Recap</h1>
    <p>
        In an exciting turn of events at the 2024 GlobalTech Innovation Summit, keynote speaker
        <strong>Mr. Robert Hayes</strong> from <em>GlobalTech Innovations</em> showcased his team's vision for AI analytics.
        His bold predictions, supported by <a href="mailto:robert.hayes@globaltech.com">robert.hayes@globaltech.com</a>, were
        backed by hard data. He stated, "With our roadmap, we're achieving a 95.2% confidence rate on AI-powered solutions."
        Reach out to Robert at +1-555-284-3489 if you have questions, or visit him at 123 Innovation Drive, Silicon Valley, CA, USA,
        where he serves as <strong>Senior Solutions Architect</strong> at <strong>GlobalTech</strong>.
    </p>

    <p>
        Next up, <strong>Ms. Jennifer Li</strong> from <strong>Finexus</strong> (Lead Data Analyst) emphasized the role of
        cross-border financial systems in the future of fintech. With confidence levels reaching 80.5%, she’s clearly paving the way
        for new global systems. To dive deeper into her work, feel free to contact Jennifer via email at
        <a href="mailto:j.li@finexus.co.uk">j.li@finexus.co.uk</a>, or at +44-20-7946-0011. Her office is located at
        15 Kensington Road, London, UK.
    </p>

    <p>
        On a different note, <strong>Ethan Parker</strong> from <em>NextGen Synergy</em> (CTO) took the stage to discuss scalable
        quantum networks. "The future is all about creating connected ecosystems," he said, noting that his team is driving the
        domestic market forward with 90.0% confidence in achieving their goals. Ethan can be reached at +1-718-555-9922 or
        <a href="mailto:e.parker@nextgen.com">e.parker@nextgen.com</a>. His main office is situated at 45 East Ave, Brooklyn, NY, USA.
    </p>

    <p>
        Meanwhile, <strong>Mrs. Emily Cheng</strong> from <em>LogixConnect</em> presented her work on cybersecurity compliance.
        She outlined a strategic plan to secure multi-cloud infrastructures, reporting a confidence rate of 85.4% in meeting domestic
        compliance standards. She can be contacted at <a href="mailto:emily.cheng@logixconnect.org">emily.cheng@logixconnect.org</a> or
        via phone at +1-323-555-4545. Her office is based at 502 Industrial Lane, Los Angeles, CA, USA.
    </p>

    <p>
        During the panel discussion, <strong>Olivia Ramirez</strong> of <em>Future Horizons</em> (Head of Operations) shared her insights
        into AI integration for urban planning. Olivia, who operates internationally with 75.6% confidence in her AI deployment predictions,
        can be reached at <a href="mailto:o.ramirez@futurehorizons.io">o.ramirez@futurehorizons.io</a> or +61-3-5559-1276. She works out of
        76 Harborview Street, Melbourne, VIC, Australia.
    </p>

    <p>
        In another highlight, <strong>Dr. Michael Foster</strong> from <em>Innovatix</em> discussed machine learning advancements that his
        domestic research team has been pioneering. With a 93.1% confidence rating, his team is on track to revolutionize the field.
        Contact him at <a href="mailto:michael.foster@innovatix.ai">michael.foster@innovatix.ai</a> or at +1-650-555-7783. He is based at
        789 Quantum Park, Palo Alto, CA, USA.
    </p>

    <p>
        During the energy panel, <strong>Mr. Daniel Clark</strong>, the Chief Energy Officer at <em>GreenPower Solutions</em>, discussed AI-powered
        energy solutions. Daniel emphasized the importance of renewable energy platforms and suggested that his company’s approach has an
        88.9% confidence rate for domestic implementation. You can reach him at <a href="mailto:d.clark@greenpower.solutions">d.clark@greenpower.solutions</a>
        or +1-646-555-2837. His New York office is located at 89 West 23rd St, New York, NY, USA.
    </p>

    <p>
        Another key speaker was <strong>Sophia Evans</strong>, the Senior AI Engineer at <em>CyberdyneTech</em>. Her team is working internationally
        on cross-compatible AI systems, with an 81.3% confidence rate in success. Get in touch with Sophia at
        <a href="mailto:sophia.evans@cyberdynetech.co">sophia.evans@cyberdynetech.co</a> or call +44-113-555-7612. Her office is located at
        52 Leeds Ave, Leeds, UK.
    </p>

    <p>
        Finally, <strong>Liam Turner</strong>, Director of Innovation at <em>StellarConnect</em>, talked about AI's role in maritime logistics.
        His international team is confident (89.4%) that their new platform will streamline operations globally. Reach out to Liam via email at
        <a href="mailto:liam.turner@stellarconnect.org">liam.turner@stellarconnect.org</a> or call +61-2-5556-9876. His main office is in
        32 Circular Quay, Sydney, NSW, Australia.
    </p>
</body>
</html>
"""

@pytest.mark.parametrize("input", [
    contacts_input_1,
    contacts_input_2
])
@base_test_llms_all
def test_evalutate_contacts(
    provider: ProviderName, 
    init_kwargs: dict, 
    func_kwargs: dict,
    input: str,
    ):

    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])

    class Contact(BaseModel):
        name: str
        """Name of the contact."""
        email: str
        """Email of the contact."""
        phone: str
        """Phone number of the contact."""
        address: str
        """Address of the contact."""
        job_title: str
        """Job title of the contact."""
        company: str
        """Company of the contact."""
        is_domestic: bool
        """True if the contact is domestic, False if international."""
        gender: Literal["Male", "Female", "Other"]    
        confidence: float
        """A confidence score for the contact information."""

        def get_name_and_email(self):
            return f"{self.name} <{self.email}>"

    class ContactsList(BaseModel):
        contacts_list: list[Contact]
        
        def print_contacts(self):
            for contact in self.contacts_list:
                print(f"Name: {contact.name}\nEmail: {contact.email}\nPhone: {contact.phone}\nAddress: {contact.address}\nJob Title: {contact.job_title}\nCompany: {contact.company}\nDomestic: {contact.is_domestic}\nGender: {contact.gender}\nConfidence: {contact.confidence}\n")


    extract_contacts = ai.get_function(FunctionConfig(
            name="extractContacts",
            system_prompt="You extract contact information from unstructued content.",
            output_parser=ContactsList,
    ))
    contacts_list = extract_contacts(input)
    assert contacts_list
    assert isinstance(contacts_list, ContactsList)
    contacts_list.print_contacts()

    print(contacts_list.contacts_list[0].get_name_and_email())



