# CHANGELOG:
# - changed lookups entry node to use long text field
# - use Union to support single lookups node or collection of nodes (restored functionality)
# - 'template' / 'negative' keys will now override existing 'templates'/'negatives' when present
# - resolutions by id during template expansion, when present as {tagname:id}
# - resolutions field passable between fields

import json
import random
import re
import yaml
from os import listdir
from os.path import exists, isdir
from os.path import join as path_join
from os.path import splitext as path_splitext
from typing import Union

from pydantic import validator

from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
    invocation_output,
    InputField,
    OutputField,
    UIComponent
)


@invocation_output("lookups_output")
class LookupTableOutput(BaseInvocationOutput):
    """Base class for invocations that output a JSON lookup table"""

    lookups: str = OutputField(description="The output lookup table")


@invocation_output("halved_prompt_output")
class HalvedPromptOutput(BaseInvocationOutput):
    """Base class for invocations that return a prompt as well as each half independently"""

    prompt: str = OutputField(default="", description="The output prompt")
    part_a: str = OutputField(default="", description="First part of the output prompt")
    part_b: str = OutputField(default="", description="Second part of the output prompt")
    resolutions: str = OutputField(default="", description="JSON dict of [tagname,id] resolutions")


@invocation(
    "lookup_table_from_file",
    title="Lookup Table from File",
    tags=["prompt", "lookups", "grammar", "file"],
    category="prompt",
    version="1.2.0",
)
class LookupTableFromFileInvocation(BaseInvocation):
    """Loads a lookup table from a YAML file"""

    file_path: str = InputField(description="Path to lookup table YAML file")

    @validator("file_path")
    def file_path_exists(cls, v):
        if not exists(v):
            raise ValueError(FileNotFoundError)
        return v

    def parsedTemplateLines(self, lines: list):
        "Parse the rows of a template key, creating copies etc."
        results = []
        for line in lines:
            if isinstance(line, str):
                if (0 < len(line)) and (line[0] == "\\"):  # We use backslash to escape parsing
                    results.append(line[1:])
                else:
                    # - N * ..., where N is a number, adds N copies of prompt "..."
                    _match = re.match(r"\s*(\d+)\s*\*\s*(.*)", line)
                    if _match is not None:
                        for i in range(int(_match[1])):
                            results.append(_match[2])
                    # - ?N, where N is a number, is treated as an instruction to add N "empty" choices:
                    elif re.match(r"\?[\d]+$", line):
                        results.extend(["" for number_of_times in range(int(line[1:]))])
                    else:
                        results.append(line)
            elif isinstance(line, list) and (len(line) == 3):
                for i in range(int(line[0])):
                    results.append(line[1:])
#            elif isinstance(line, list):
#                results.append(line)
            else:
                results.append(line)
        return results

    def lookupTableFromFile(self, file_path: str):
        lookup_table = {}
        raw_conf = ''
        with open(file_path, 'r') as inf:
            raw_conf = yaml.safe_load(inf)
        prompt_section = raw_conf.get("prompt")
        if isinstance(prompt_section, dict):
            for k, v in prompt_section.items():
                # Add singular "template" and "negative" entries
                #  to "templates" and "negatives" lists:
                if k == "template":
                    if "templates" not in lookup_table:
                        lookup_table["templates"] = []
                    lookup_table["templates"].extend(self.parsedTemplateLines([v]))
                elif k == "negative":
                    if "negatives" not in lookup_table:
                        lookup_table["negatives"] = []
                    lookup_table["negatives"].extend(self.parsedTemplateLines([v]))
                else:
                    if k not in lookup_table:
                        lookup_table[k] = []
                    lookup_table[k].extend(self.parsedTemplateLines(v))
        return lookup_table

    def invoke(self, context: InvocationContext) -> LookupTableOutput:
        lookups = None
        if isdir(self.file_path):
            lookups = {}
            for pathname in listdir(self.file_path):
                if (not isdir(pathname)) and (path_splitext(pathname)[-1].lower() == ".yaml"):
                    pathname = path_join(self.file_path, pathname)
                    this_lookup_table = self.lookupTableFromFile(pathname)
                    for k, v in iter(this_lookup_table.items()):
                        if k not in lookups:
                            lookups[k] = []
                        lookups[k].extend(v)
        else:
            lookups = self.lookupTableFromFile(self.file_path)
        return LookupTableOutput(lookups=json.dumps(lookups))


@invocation(
    "lookup_from_prompt",
    title="Lookups Entry from Prompt",
    tags=["prompt", "lookups", "grammar"],
    category="prompt",
    version="1.3.2",
)
class LookupsEntryFromPromptInvocation(BaseInvocation):
    """Creates a lookup table of a single heading->value"""

    heading: str = InputField(description="Heading for the lookup table entry")
    lookup: str = InputField(
        default="",
        description="The entry to place under Heading in the lookup table",
        ui_component=UIComponent.Textarea,        
    )

    def invoke(self, context: InvocationContext) -> LookupTableOutput:
        lookups = {}
        lookups[self.heading] = [self.lookup]
        return LookupTableOutput(lookups=json.dumps(lookups))


@invocation(
    "prompt_from_lookup_table",
    title="Prompt from Lookup Table",
    tags=["prompt", "lookups", "grammar"],
    category="prompt",
    version="1.3.2",
    use_cache=False,
)
class PromptFromLookupTableInvocation(BaseInvocation):
    """Creates prompts using lookup table templates"""

    lookups: Union[str, list[str]] = InputField(
        description="Lookup table(s) containing template(s) (JSON)",
        default=[],
    )
    remove_negatives: bool = InputField(default=False, description="Whether to strip out text between []")
    strip_parens_probability: float = InputField(
        default=0.0, ge=0.0, le=1.0, description="Probability of removing attention group weightings"
    )
    resolutions: Union[str, list[str]] = InputField(
        description="JSON structure of substitutions by id by tag",
        default=[],
    )
    resolutions_dict: dict = InputField(
        description="Private field for id substitutions dict cache",
        ui_hidden=True,
        default={},
    )

    @validator("lookups")
    def validate_lookups(cls, v):
        valid = False
        if isinstance(v, list):
            for i in v:
                loaded = json.loads(i)
                if (("templates" in loaded) or ("template" in loaded)):
                    valid = True
                    break
        else:
            if (
                    (not (v is None)) and
                    (("templates" in json.loads(v)) or ("template" in json.loads(v)))
            ):
                valid = True
        if (not valid) and (not (len(v) == 0)):
            raise ValueError("'template' or 'templates' key must be present in lookups")
        return v

    def iterateExpansions(self, _base, _reflection, lookups):
        _next, _reflection = self.templateExpand(_base, lookups=lookups, reflection=_reflection)
        while (_next != _base) or (re.search(r"{\w+}", _base)):
            _base = _next
            _next, _reflection = self.templateExpand(_base, lookups=lookups, reflection=_reflection)
        _appendices = ""
        while _reflection != "":
            _appendix = _reflection
            _next, _reflection = self.templateExpand(_appendix, lookups=lookups, reflection="")
            while (_next != _appendix) or (re.search(r"{\w+}", _appendix)):
                _appendix = _next
                _next, _reflection = self.templateExpand(_appendix, lookups=lookups, reflection=_reflection)
            _appendices = _appendices + _appendix

        return _base, _appendices

    def templateExpand(self, s: str, lookups: dict, reflection: str = ""):
        "Used internally to replace words with their template lookups."
        _split = re.split(r"({[\d:\w]+})", s)
        result = ""
        for word in _split:
            _lookup = None

            if re.fullmatch(r"({[\d:\w]+})", word):
                # This word needs subsitution
                # Track unique identifiers for tags:
                word_id = None
                id_parts = word[1:-1].split(":")
                if 1 < len(id_parts):
                    word, word_id = tuple(id_parts[:2])
                else:
                    word = word[1:-1]
                if word_id:
                    if word in self.resolutions_dict:
                        if word_id in self.resolutions_dict[word]:
                            _lookup = self.resolutions_dict[word][word_id]
                    else:
                        self.resolutions_dict[word] = {}
                if not _lookup:
                    _lookup = random.choice(lookups[word])

                if isinstance(_lookup, (list, list)):
                    # This is a two-part substition (A, B)
                    if word_id:
                        # Expand until done so the resolution is fully cached and reproducible:
                        _base, _reflection = _lookup[:2]

                        _base, _appendices = self.iterateExpansions(_base, _reflection, lookups)
                        self.resolutions_dict[word][word_id] = [_base, _appendices]

                        result = result + _base
                        reflection = " " + _appendices + reflection
                    else:
                        result = result + _lookup[0]
                        reflection = " " + _lookup[1] + reflection
                    
                else:  # Otherwise, lookup is just a string
                    if word_id:
                        self.resolutions_dict[word][word_id] = _lookup
                        _base, _reflection = _lookup, ''

                        _base, _appendices = self.iterateExpansions(_base, _reflection, lookups)
                        self.resolutions_dict[word][word_id] = [_base, _appendices]

                        result = result + _base
                        reflection = " " + _appendices + reflection
                        
                    else:
                        result = result + _lookup

            else:  # Direct pass-through
                _lookup = word                
                if isinstance(_lookup, list):
                    # This is a two-part substition (A, B)
                    result = result + _lookup[0]
                    reflection = " " + _lookup[1] + reflection
                else:  # Direct pass-through
                    result = result + _lookup
                
        return result, reflection

    def cleanup(self, string_in: str):
        "Regex to clean formatting typos occurring during generation (TODO: improve)"
        # Condense whitespace
        _str = re.sub(r"\s+", " ", string_in, 0)
        _str = re.sub(r"\s([,;])\s", lambda match: (match.group(1) + " "), _str, 0)
        # Remove empty sets of parens
        _str = re.sub(r"\(\)[\+\-]*", "", _str)
        # Condense whitespace again
        _str = re.sub(r"\s+", " ", _str, 0)
        _str = re.sub(r"\s([,;])\s", lambda match: (match.group(1) + " "), _str, 0)
        _str = re.sub(r",+\s", ", ", _str, 0)
        # Now we attempt to combine prefixes and suffixes.
        _collapsed = re.sub(
            r"(\S+?)\s*@@\s*(\S+)", lambda match: self.prefixer(match.group(1), match.group(2)), _str, 0
        )
        while _collapsed != _str:
            _str = _collapsed
            _collapsed = re.sub(
                r"(\S+?)\s*@@\s*(\S+)", lambda match: self.prefixer(match.group(1), match.group(2)), _str, 0
            )
        # Remove instances of "a apple" errors by subbing "an", w/ or w/o parens in the way.
        string_out = re.sub(
            r"(^[Aa]\s|\s[Aa]\s)([\(]*)([aeiouAEIOU])",
            lambda match: ((" a" if (len(match.group(1)) == 3) else "a") + "n " + match.group(2) + match.group(3)),
            _str,
            0,
        )
        return string_out

    def prefixer(self, a: str, b: str):
        "Attach pre/suffixes (indicated w/'@@') and rearrange parens/+/-"
        result = None
        if re.match(r"[\)\+\-]+$", b):
            result = a + b + " @@"
        else:
            # Allow a single trailing hyphen but move the rest of ), +, -:
            _aWeighting = re.match(r"\S*?(-?)([\)*\+\-]*)\s*$", a)
            _newA = a
            if _aWeighting is not None:
                _newA = a[0 : _aWeighting.start(2)]
                _aWeighting = _aWeighting[2]
            # grab parens from the front of b:
            _bWeighting = re.match(r"\s*(\(*)(\S*)", b)
            _newB = b
            if _bWeighting is not None:
                _newB = _bWeighting[2]
                _bWeighting = _bWeighting[1]
            # now subtract the parens that cancel out:
            _newAWeighting = _aWeighting.replace(")", "", _bWeighting.count("("))
            _nSubs = len(_aWeighting) - len(_newAWeighting)
            _newBWeighting = _bWeighting.replace("(", "", _nSubs) if (0 < _nSubs) else _bWeighting
            _raw = _newBWeighting + _newA + _newB + _newAWeighting
            # subtract trailing +-'s that cancel:
            result = _raw.replace("+-", "", 1)
            while _raw != result:
                _raw = result
                result = _raw.replace("+-", "", 1)
        return result

    def invoke(self, context: InvocationContext) -> HalvedPromptOutput:
        resolutions_dict = {}
        if isinstance(self.resolutions, list):
            for resolutions_table in reversed(self.resolutions):
                resolutions_table = json.loads(resolutions_table)
                for k, v in iter(resolutions_table.items()):
                    if k not in resolutions_dict:
                        resolutions_dict[k] = []
                    resolutions_dict[k].extend(v)
        else:
            resolutions_dict = json.loads(self.resolutions)
        self.resolutions_dict = resolutions_dict
        lookups = {}
        if isinstance(self.lookups, list):
            for lookup_table in reversed(self.lookups):
                lookup_table = json.loads(lookup_table)
                for k, v in iter(lookup_table.items()):
                    if k not in lookups:
                        lookups[k] = []
                    lookups[k].extend(v)
        else:
            lookups = json.loads(self.lookups)
        template_strings = (
            lookups['template'] if 'template' in lookups else lookups["templates"]
        )
        base_negatives = (
            lookups['negative'] if ('negative' in lookups) else (
                lookups["negatives"] if ("negatives" in lookups) else []
            )
        )
        result = None
        appendices = None

        for i in range(1):
            base, reflection = self.templateExpand(
                random.choice(template_strings),
                lookups=lookups,
                reflection=""
            )
            base, appendices = self.iterateExpansions(base, reflection, lookups)

            if random.random() < self.strip_parens_probability:
                # Strip off parentheses and pluses, then trailing minuses w/o and w/ commas afterward
                base = re.sub(r"[\(\)+]", "", base, 0)
                base = re.sub(r"\-+\s", " ", base, 0)
                base = re.sub(r"\-+,", ",", base, 0)
                appendices = re.sub(r"[\(\)+]", "", appendices, 0)
                appendices = re.sub(r"\-+\s", " ", appendices, 0)
                appendices = re.sub(r"\-+,", ",", appendices, 0)
            if self.remove_negatives:
                # strip out anything between []
                base = re.sub(r"\[[^\]\[]*\]", "", base, 0)
                appendices = re.sub(r"\[[^\]\[]*\]", "", appendices, 0)

            base = self.cleanup(base)
            appendices = self.cleanup(appendices)

            if not (self.remove_negatives or (not base_negatives)):
                appendices = appendices + " " + random.choice(base_negatives)

            result = (base + appendices).strip()

        return HalvedPromptOutput(
            prompt=result,
            part_a=base.strip(),
            part_b=appendices.strip(),
            resolutions=json.dumps(self.resolutions_dict)
        )
