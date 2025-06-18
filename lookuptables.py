import json
import random
import re
import sys
import yaml
from os import listdir
from os.path import exists, isdir
from os.path import join as path_join
from os.path import splitext as path_splitext
from typing import Any, Optional, Union, List, Dict, Tuple

from pydantic import validator

from invokeai.backend.util.logging import InvokeAILogger
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

_logger = InvokeAILogger.get_logger(__name__)


@invocation_output("lookups_output")
class LookupTableOutput(BaseInvocationOutput):
    """Base class for invocations that output a JSON lookup table"""

    lookups: str = OutputField(description="The output lookup table")


@invocation_output("halved_prompt_output")
class HalvedPromptOutput(BaseInvocationOutput):
    """Output for a halved prompt."""
    prompt: str = OutputField(default="", description="The entire output prompt")
    part_a: str = OutputField(default="", description="The first part of the output prompt")
    part_b: str = OutputField(default="", description="The second part of the output prompt")
    resolutions: str = OutputField(default="", description="JSON dict of [tagname,id] resolutions")
    seed_vector: str = OutputField(default="", description="JSON string of the seed vector used for generation")


@invocation_output("evolutionary_prompt_list_output")
class EvolutionaryPromptListOutput(BaseInvocationOutput):
    """Output for a list of generated prompts and their seed vectors."""
    # Each string in the list will be a JSON array: `[prompt_string, seed_vector_json_string]`
    prompt_seed_pairs: list[str] = OutputField(description="List of JSON strings, each containing [prompt, seed_vector_json]")
    selected_pair: str = OutputField(description="The index-selected prompt from the output population")

    
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
    version="1.5.0",
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
    seed_vector_in: Optional[str] = InputField(
        default=None, description="Optional JSON array of seeds for deterministic generation"
    )

    @validator("lookups")
    def validate_lookups(cls, v):
        valid = False
        if isinstance(v, list):
            for i in v:
                loaded = json.loads(i)
                if ("templates" in loaded) or ("template" in loaded):
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

    # --- Internal Helper for Seed Management ---
    class SeedManager:
        def __init__(self, seed_vector_in: Optional[list] = None):
            self.seed_vector_in = seed_vector_in if seed_vector_in is not None else []
            self.seed_vector_out = []
            self.current_seed_index = 0

        def get_choice(self, options_list: list) -> Any:
            """
            Gets a choice using the next seed from the input vector if available,
            otherwise generates a new random seed and appends it to output vector.
            """
            num_options = len(options_list)
            
            if num_options == 0:
                raise ValueError("Cannot choose from an empty list of options.")

            if self.current_seed_index < len(self.seed_vector_in):
                # Use provided seed
                seed_val = self.seed_vector_in[self.current_seed_index]
                self.current_seed_index += 1
            else:
                # Generate new random seed if input vector is exhausted
                seed_val = random.randint(0, sys.maxsize)
            
            # Store the seed that was actually used for this choice
            self.seed_vector_out.append(seed_val)
            
            # Deterministically map the seed to an option
            # Using modulus to ensure the index is within bounds of options_list
            choice_index = seed_val % num_options
            return options_list[choice_index]

    def iterateExpansions(self, _base, _reflection, lookups, seed_manager: "SeedManager"):
        _next, _reflection = self.templateExpand(_base, lookups=lookups, reflection=_reflection, seed_manager=seed_manager)
        while (_next != _base) or (re.search(r"{\w+}", _base)):
            _base = _next
            _next, _reflection = self.templateExpand(_base, lookups=lookups, reflection=_reflection, seed_manager=seed_manager)
        _appendices = ""
        while _reflection != "":
            _appendix = _reflection
            _next, _reflection = self.templateExpand(_appendix, lookups=lookups, reflection="", seed_manager=seed_manager)
            while (_next != _appendix) or (re.search(r"{\w+}", _appendix)):
                _appendix = _next
                _next, _reflection = self.templateExpand(_appendix, lookups=lookups, reflection=_reflection, seed_manager=seed_manager)
            _appendices = _appendices + _appendix

        return _base, _appendices

    def templateExpand(self, s: str, lookups: dict, reflection: str = "", seed_manager: "SeedManager" = None):
        "Used internally to replace words with their template lookups."
        if seed_manager is None:
            # Fallback for direct calls not through invoke or iterateExpansions if needed
            seed_manager = self.SeedManager() 
            
        _split = re.split(r"({[\d:\w]+})", s)
        result = ""
        for word in _split:
            _lookup = None

            if re.fullmatch(r"({[\d:\w]+})", word):
                # This word needs substitution
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
                    # Use the seed manager for choice instead of random.choice
                    _lookup = seed_manager.get_choice(lookups[word])

                if isinstance(_lookup, list):
                    # This is a two-part substitution (A, B)
                    if word_id:
                        # Expand until done so the resolution is fully cached and reproducible:
                        _base, _reflection = _lookup[:2]

                        _base, _appendices = self.iterateExpansions(_base, _reflection, lookups, seed_manager) # Pass seed_manager
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

                        _base, _appendices = self.iterateExpansions(_base, _reflection, lookups, seed_manager) # Pass seed_manager
                        self.resolutions_dict[word][word_id] = [_base, _appendices]

                        result = result + _base
                        reflection = " " + _appendices + reflection
                        
                    else:
                        result = result + _lookup

            else:  # Direct pass-through
                _lookup = word                
                if isinstance(_lookup, list):
                    # This is a two-part substitution (A, B)
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
            _result = _raw.replace("+-", "", 1)
            while _raw != _result:
                _raw = _result
                _result = _raw.replace("+-", "", 1)
            result = _result # Assign the final result back to 'result'
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
            # Parse and store the lookup tables with a sortable key
            parsed_lookups = []
            for lookup_table_str in self.lookups:
                parsed_table = json.loads(lookup_table_str)
                # Create a sortable representation: tuple of sorted (key, value) pairs
                # This makes the sorting deterministic based on content
                sort_key = tuple(sorted(parsed_table.items()))
                parsed_lookups.append((sort_key, parsed_table))

            # Sort the parsed lookup tables deterministically
            parsed_lookups.sort(key=lambda x: x[0])

            for sort_key, lookup_table in reversed(parsed_lookups):
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

        # Initialize SeedManager
        initial_seed_vector = None
        if self.seed_vector_in:
            initial_seed_vector = json.loads(self.seed_vector_in)
        seed_manager = self.SeedManager(initial_seed_vector)

        for i in range(1):
            # Use seed_manager for the initial template choice
            initial_template = seed_manager.get_choice(template_strings)
            
            base, reflection = self.templateExpand(
                initial_template,
                lookups=lookups,
                reflection="",
                seed_manager=seed_manager # Pass the seed_manager
            )
            base, appendices = self.iterateExpansions(base, reflection, lookups, seed_manager) # Pass the seed_manager

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
                # Use seed_manager for negative choice as well
                appendices = appendices + " " + seed_manager.get_choice(base_negatives)

            result = (base + appendices).strip()

        # Output the generated seed vector
        generated_seed_vector_json = json.dumps(seed_manager.seed_vector_out)

        return HalvedPromptOutput(
            prompt=result,
            part_a=base.strip(),
            part_b=appendices.strip(),
            resolutions=json.dumps(self.resolutions_dict),
            seed_vector=generated_seed_vector_json # Add seed_vector to output
        )

class SeedManager:
    """
    Manages seed consumption and output, with added branch tracking for crossover.
    """
    class SeedBranchTracker:
        """Tracks the seed consumption for a specific branch (non-terminal expansion)."""
        def __init__(self, non_terminal_name: str, start_index: int):
            self.non_terminal_name = non_terminal_name
            self.start_seed_index = start_index
            self.end_seed_index: Optional[int] = None # Will be set once the branch is fully expanded

    def __init__(
        self, 
        initial_seed_vector: Optional[list] = None,
        crossover_target_nt: Optional[str] = None,
        crossover_subvector: Optional[list] = None,
        crossover_insertion_point: Optional[int] = None
    ):
        self.seed_vector_in = initial_seed_vector if initial_seed_vector is not None else []
        self.seed_vector_out = [] # This accumulates the seeds actually used for the current generation
        self.current_seed_index = 0
        
        # For crossover logic
        self.crossover_target_nt = crossover_target_nt # The non-terminal whose expansion we want to swap
        self.crossover_subvector = crossover_subvector # The seeds to insert for the crossover target
        self.crossover_insertion_point = crossover_insertion_point # The index in seed_vector_in where crossover starts

        self.in_crossover_branch = False # Flag to indicate if we're currently processing the crossover branch
        self.crossover_subvector_index = 0 # Index within the crossover_subvector

        self.branch_traces: List[SeedManager.SeedBranchTracker] = []
        self._current_branch_stack: List[SeedManager.SeedBranchTracker] = [] # To manage nested branches

    def start_branch(self, non_terminal_name: str):
        """Called when a new non-terminal expansion begins."""
        tracker = self.SeedBranchTracker(non_terminal_name, self.current_seed_index)
        self.branch_traces.append(tracker)
        self._current_branch_stack.append(tracker)

        if self.crossover_target_nt and not self.in_crossover_branch:
            # Check if this is the target non-terminal for crossover AND it's the first one encountered
            if non_terminal_name == self.crossover_target_nt:
                _logger.debug(f"Entered crossover branch for {non_terminal_name} at seed index {self.current_seed_index}")
                self.in_crossover_branch = True
                # Skip the original seeds in self.seed_vector_in for this branch
                # The actual skipping will be handled by get_choice logic

    def end_branch(self):
        """Called when a non-terminal expansion finishes."""
        if self._current_branch_stack:
            tracker = self._current_branch_stack.pop()
            tracker.end_seed_index = self.current_seed_index
        
        # If we're exiting the crossover branch (and it was the one we entered)
        if self.in_crossover_branch and not self._current_branch_stack: # Ensure we've left all nested calls that were part of the crossover branch
            _logger.debug(f"Exited crossover branch at seed index {self.current_seed_index}")
            self.in_crossover_branch = False
            self.crossover_subvector_index = 0 # Reset for next potential crossover, though unlikely in a single run

    def get_choice(self, options_list: list) -> Any:
        """
        Gets a choice using the next seed. Handles input vector, crossover subvector,
        and new random generation.
        """
        num_options = len(options_list)
        
        if num_options == 0:
            raise ValueError("Cannot choose from an empty list of options.")

        seed_val = None

        if self.in_crossover_branch and self.crossover_subvector is not None:
            # We are in the designated crossover branch, use seeds from the crossover_subvector
            if self.crossover_subvector_index < len(self.crossover_subvector):
                seed_val = self.crossover_subvector[self.crossover_subvector_index]
                self.crossover_subvector_index += 1
                # We still increment current_seed_index to correctly track the original position
                # where these seeds *would have been* if they were from the original parent.
                # This helps in defining the end of the "skipped" section.
                self.current_seed_index += 1 
                _logger.debug(f"Using CO seed: {seed_val} (idx: {self.crossover_subvector_index-1}) for {options_list[0] if options_list else 'UNKNOWN'} (next original seed: {self.current_seed_index})")
            else:
                # Crossover subvector exhausted, generate new random seeds for the rest of this branch
                seed_val = random.randint(0, sys.maxsize)
                self.current_seed_index += 1 # Continue to increment original seed index
                _logger.debug(f"CO subvector exhausted, generating NEW random seed: {seed_val} (next original seed: {self.current_seed_index})")
        else:
            # Not in crossover branch, or crossover not active for this generation.
            # Use original input seeds or generate new ones.
            if self.current_seed_index < len(self.seed_vector_in):
                seed_val = self.seed_vector_in[self.current_seed_index]
                self.current_seed_index += 1
                _logger.debug(f"Using original IN seed: {seed_val} (idx: {self.current_seed_index-1}) for {options_list[0] if options_list else 'UNKNOWN'}")
            else:
                # Input seed vector exhausted, generate new random seeds
                seed_val = random.randint(0, sys.maxsize)
                self.current_seed_index += 1
                _logger.debug(f"Original IN vector exhausted, generating NEW random seed: {seed_val} (idx: {self.current_seed_index-1})")
        
        # Store the seed that was actually used for this choice (for the output seed_vector)
        self.seed_vector_out.append(seed_val)
        
        # Deterministically map the seed to an option
        choice_index = seed_val % num_options
        return options_list[choice_index]

class _BasePromptGenerator:
    """
    A base class to hold shared prompt generation logic like templateExpand, iterateExpansions, cleanup.
    This helps avoid code duplication between PromptFromLookupTableInvocation and CrossoverPromptInvocation.
    """
    resolutions_dict: dict # This would typically be passed around or handled by the node's state.

    def iterateExpansions(self, _base, _reflection, lookups, seed_manager: SeedManager):
        # We need to correctly pass the non-terminal name to start_branch for initial template and subsequent expansions.
        # This current setup doesn't directly expose the non-terminal name from iterateExpansions,
        # but templateExpand will call start_branch with the correct name.
        _next, _reflection = self.templateExpand(_base, lookups=lookups, reflection=_reflection, seed_manager=seed_manager)
        while (_next != _base) or (re.search(r"{\w+}", _base)):
            _base = _next
            _next, _reflection = self.templateExpand(_base, lookups=lookups, reflection=_reflection, seed_manager=seed_manager)
        _appendices = ""
        while _reflection != "":
            _appendix = _reflection
            _next, _reflection = self.templateExpand(_appendix, lookups=lookups, reflection="", seed_manager=seed_manager)
            while (_next != _appendix) or (re.search(r"{\w+}", _appendix)):
                _appendix = _next
                _next, _reflection = self.templateExpand(_appendix, lookups=lookups, reflection=_reflection, seed_manager=seed_manager)
            _appendices = _appendices + _appendix

        return _base, _appendices

    def templateExpand(self, s: str, lookups: dict, reflection: str = "", seed_manager: SeedManager = None):
        "Used internally to replace words with their template lookups."
        if seed_manager is None:
            seed_manager = SeedManager() 
            
        _split = re.split(r"({[\d:\w]+})", s)
        result = ""
        for word_raw in _split:
            _lookup = None

            if re.fullmatch(r"({[\d:\w]+})", word_raw):
                # This word needs substitution
                # Track unique identifiers for tags:
                word_id = None
                id_parts = word_raw[1:-1].split(":")
                if 1 < len(id_parts):
                    word_key, word_id = tuple(id_parts[:2])
                else:
                    word_key = word_raw[1:-1]
                
                # --- Start tracking this non-terminal's branch ---
                # This is a bit tricky as `templateExpand` handles individual words,
                # not the overall non-terminal *rule* itself.
                # However, for our crossover, we need to track the *start of expansion* for a given non-terminal key.
                # The assumption here is that `word_key` directly maps to a non-terminal in `lookups`.
                
                # Check if `word_key` is in lookups, meaning it's a non-terminal that will be expanded
                if word_key in lookups:
                    seed_manager.start_branch(word_key) # Signal start of this non-terminal's branch
                # --- End tracking setup ---

                if word_id:
                    if word_key in self.resolutions_dict:
                        if word_id in self.resolutions_dict[word_key]:
                            _lookup = self.resolutions_dict[word_key][word_id]
                    else:
                        self.resolutions_dict[word_key] = {}
                if not _lookup:
                    # Use the seed manager for choice instead of random.choice
                    _lookup = seed_manager.get_choice(lookups[word_key])

                if isinstance(_lookup, list):
                    # This is a two-part substitution (A, B)
                    if word_id:
                        # Expand until done so the resolution is fully cached and reproducible:
                        _base, _reflection = _lookup[:2]

                        _base, _appendices = self.iterateExpansions(_base, _reflection, lookups, seed_manager) # Pass seed_manager
                        self.resolutions_dict[word_key][word_id] = [_base, _appendices]

                        result = result + _base
                        reflection = " " + _appendices + reflection
                    else:
                        result = result + _lookup[0]
                        reflection = " " + _lookup[1] + reflection
                    
                else:  # Otherwise, lookup is just a string
                    if word_id:
                        self.resolutions_dict[word_key][word_id] = _lookup
                        _base, _reflection = _lookup, ''

                        _base, _appendices = self.iterateExpansions(_base, _reflection, lookups, seed_manager) # Pass seed_manager
                        self.resolutions_dict[word_key][word_id] = [_base, _appendices]

                        result = result + _base
                        reflection = " " + _appendices + reflection
                        
                    else:
                        result = result + _lookup
                
                # --- End tracking this non-terminal's branch ---
                if word_key in lookups:
                    seed_manager.end_branch() # Signal end of this non-terminal's branch

            else:  # Direct pass-through (not a substitution)
                _lookup = word_raw                
                if isinstance(_lookup, list): # Should not happen for a direct pass-through unless it's a pre-processed list
                    result = result + _lookup[0]
                    reflection = " " + _lookup[1] + reflection
                else:
                    result = result + _lookup
                
        return result, reflection

    def cleanup(self, string_in: str):
        "Regex to clean formatting typos occurring during generation (TODO: improve)"
        _str = re.sub(r"\s+", " ", string_in, 0)
        _str = re.sub(r"\s([,;])\s", lambda match: (match.group(1) + " "), _str, 0)
        _str = re.sub(r"\(\)[\+\-]*", "", _str)
        _str = re.sub(r"\s+", " ", _str, 0)
        _str = re.sub(r"\s([,;])\s", lambda match: (match.group(1) + " "), _str, 0)
        _str = re.sub(r",+\s", ", ", _str, 0)
        _collapsed = re.sub(
            r"(\S+?)\s*@@\s*(\S+)", lambda match: self.prefixer(match.group(1), match.group(2)), _str, 0
        )
        while _collapsed != _str:
            _str = _collapsed
            _collapsed = re.sub(
                r"(\S+?)\s*@@\s*(\S+)", lambda match: self.prefixer(match.group(1), match.group(2)), _str, 0
            )
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
            _aWeighting = re.match(r"\S*?(-?)([\)*\+\-]*)\s*$", a)
            _newA = a
            if _aWeighting is not None:
                _newA = a[0 : _aWeighting.start(2)]
                _aWeighting = _aWeighting[2]
            _bWeighting = re.match(r"\s*(\(*)(\S*)", b)
            _newB = b
            if _bWeighting is not None:
                _newB = _bWeighting[2]
                _bWeighting = _bWeighting[1]
            _newAWeighting = _aWeighting.replace(")", "", _bWeighting.count("("))
            _nSubs = len(_aWeighting) - len(_newAWeighting)
            _newBWeighting = _bWeighting.replace("(", "", _nSubs) if (0 < _nSubs) else _bWeighting
            _raw = _newBWeighting + _newA + _newB + _newAWeighting
            _result = _raw.replace("+-", "", 1)
            while _raw != _result:
                _raw = _result
                _result = _raw.replace("+-", "", 1)
            result = _result
        return result

@invocation(
    "crossover_prompt",
    title="Crossover Prompt",
    tags=["prompt", "lookups", "grammar", "crossover", "GA"],
    category="prompt",
    version="1.4.1", # Increment version from base node
    use_cache=False,
)
class CrossoverPromptInvocation(_BasePromptGenerator, BaseInvocation):
    """
    Performs a crossover operation on two parent seed vectors to generate a new prompt.
    """

    # --- Standard fields from PromptFromLookupTableInvocation ---
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

    # --- Crossover specific inputs ---
    parent_a_seed_vector_in: str = InputField(
        description="JSON array of seeds for Parent A's generation",
    )
    parent_b_seed_vector_in: str = InputField(
        description="JSON array of seeds for Parent B's generation",
    )
    child_a_or_b: bool = InputField(
        default=True,
        description="True for Child A (Parent A + B's branch), False for Child B (Parent B + A's branch)",
    )
    # This input will allow the user to specify the non-terminal to target for crossover.
    # It assumes the lookup tables contain keys that correspond to non-terminals.
    crossover_non_terminal: Optional[str] = InputField(
        default=None,
        description="Optional: The non-terminal (key in lookups) to target for the crossover branch. If None, a random one will be chosen from available non-terminals.",
    )


    @validator("lookups")
    def validate_lookups(cls, v):
        valid = False
        if isinstance(v, list):
            for i in v:
                loaded = json.loads(i)
                if ("templates" in loaded) or ("template" in loaded):
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

    def invoke(self, context: InvocationContext) -> HalvedPromptOutput:
        # Initialize resolutions_dict for this instance (from _BasePromptGenerator)
        self.resolutions_dict = {} 

        # --- Load and prepare lookups and resolutions ---
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
        self.resolutions_dict = resolutions_dict # Store in instance

        lookups = {}
        if isinstance(self.lookups, list):
            # Parse and store the lookup tables with a sortable key
            parsed_lookups = []
            for lookup_table_str in self.lookups:
                parsed_table = json.loads(lookup_table_str)
                # Create a sortable representation: tuple of sorted (key, value) pairs
                # This makes the sorting deterministic based on content
                sort_key = tuple(sorted(parsed_table.items()))
                parsed_lookups.append((sort_key, parsed_table))

            # Sort the parsed lookup tables deterministically
            parsed_lookups.sort(key=lambda x: x[0])

            for sort_key, lookup_table in reversed(parsed_lookups):
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
        
        # --- Crossover Logic Setup ---
        parent_a_seeds = json.loads(self.parent_a_seed_vector_in)
        parent_b_seeds = json.loads(self.parent_b_seed_vector_in)

        # 1. First, expand Parent A and Parent B fully to "trace" their branches.
        # This is necessary to identify the start/end indices of the subvectors for crossover.
        # We'll use temporary SeedManagers for this tracing.

        # Trace Parent A
        temp_seed_manager_a = SeedManager(parent_a_seeds)
        # Dummy run to get the trace information
        # Need to re-run the whole generation process to get the trace
        self._run_dummy_generation(template_strings, lookups, base_negatives, temp_seed_manager_a)
        parent_a_branch_traces = temp_seed_manager_a.branch_traces
        
        # Trace Parent B
        temp_seed_manager_b = SeedManager(parent_b_seeds)
        self._run_dummy_generation(template_strings, lookups, base_negatives, temp_seed_manager_b)
        parent_b_branch_traces = temp_seed_manager_b.branch_traces

        # 2. Identify common non-terminals that can be targeted for crossover.
        # Collect all unique non-terminal names that appear as start points for branches.
        # Ensure we only pick from non-terminals that actually generated a branch (i.e., have start and end indices)
        valid_a_nts = {t.non_terminal_name for t in parent_a_branch_traces if t.end_seed_index is not None}
        valid_b_nts = {t.non_terminal_name for t in parent_b_branch_traces if t.end_seed_index is not None}
        
        # Include 'template' as a potential non-terminal if it was used to start the generation
        if temp_seed_manager_a.seed_vector_out: # Check if anything was generated
             # The first choice is always for 'template'
            valid_a_nts.add('template') 
        if temp_seed_manager_b.seed_vector_out:
            valid_b_nts.add('template')
            
        common_nts = list(valid_a_nts.intersection(valid_b_nts))

        if not common_nts:
            _logger.warning("No common non-terminals found for crossover between parents. Generating child A from Parent A, Child B from Parent B.")
            # Fallback: if no common points, just return one of the parents as the child
            if self.child_a_or_b:
                generated_seed_vector_json = json.dumps(parent_a_seeds) # No crossover applied
                return self._generate_prompt_from_seeds(parent_a_seeds, template_strings, lookups, base_negatives)
            else:
                generated_seed_vector_json = json.dumps(parent_b_seeds) # No crossover applied
                return self._generate_prompt_from_seeds(parent_b_seeds, template_strings, lookups, base_negatives)

        # 3. Choose the crossover non-terminal.
        target_non_terminal = self.crossover_non_terminal
        if target_non_terminal is None or target_non_terminal not in common_nts:
            _logger.info(f"Crossover non-terminal not specified or not valid. Choosing random from: {common_nts}")
            target_non_terminal = random.choice(common_nts)
        
        _logger.info(f"Selected crossover non-terminal: {target_non_terminal}")

        # 4. Find the specific branch subvectors for the chosen non-terminal.
        # For simplicity, we'll pick the *first* occurrence of the target non-terminal in each trace.
        branch_a_tracker: Optional[SeedManager.SeedBranchTracker] = None
        for t in parent_a_branch_traces:
            if t.non_terminal_name == target_non_terminal and t.end_seed_index is not None:
                branch_a_tracker = t
                break
        
        branch_b_tracker: Optional[SeedManager.SeedBranchTracker] = None
        for t in parent_b_branch_traces:
            if t.non_terminal_name == target_non_terminal and t.end_seed_index is not None:
                branch_b_tracker = t
                break
        
        if not (branch_a_tracker and branch_b_tracker):
             _logger.warning(f"Could not find valid branches for '{target_non_terminal}' in both parents. Generating child A from Parent A, Child B from Parent B.")
             if self.child_a_or_b:
                generated_seed_vector_json = json.dumps(parent_a_seeds) # No crossover applied
                return self._generate_prompt_from_seeds(parent_a_seeds, template_strings, lookups, base_negatives)
             else:
                generated_seed_vector_json = json.dumps(parent_b_seeds) # No crossover applied
                return self._generate_prompt_from_seeds(parent_b_seeds, template_strings, lookups, base_negatives)


        # Extract the subvectors
        # Note: end_seed_index points to the index *after* the last used seed.
        subvector_a = parent_a_seeds[branch_a_tracker.start_seed_index : branch_a_tracker.end_seed_index]
        subvector_b = parent_b_seeds[branch_b_tracker.start_seed_index : branch_b_tracker.end_seed_index]

        _logger.debug(f"Parent A subvector for '{target_non_terminal}': {subvector_a} (indices {branch_a_tracker.start_seed_index}-{branch_a_tracker.end_seed_index-1})")
        _logger.debug(f"Parent B subvector for '{target_non_terminal}': {subvector_b} (indices {branch_b_tracker.start_seed_index}-{branch_b_tracker.end_seed_index-1})")

        # 5. Construct the offspring's initial seed vector for the *real* generation.
        offspring_initial_seed_vector: List[int] = []
        crossover_subvector_to_use: Optional[List[int]] = None
        crossover_insertion_point: Optional[int] = None

        if self.child_a_or_b: # Generate Child A (Parent A + B's branch)
            _logger.info("Generating Child A (Parent A + B's branch)")
            offspring_initial_seed_vector.extend(parent_a_seeds[:branch_a_tracker.start_seed_index])
            crossover_subvector_to_use = subvector_b
            crossover_insertion_point = branch_a_tracker.start_seed_index
            # The rest of Parent A's seeds will be implicitly handled by the SeedManager,
            # which will skip the original Parent A's branch seeds and then resume.
            offspring_initial_seed_vector.extend(parent_a_seeds[branch_a_tracker.end_seed_index:])
            
        else: # Generate Child B (Parent B + A's branch)
            _logger.info("Generating Child B (Parent B + A's branch)")
            offspring_initial_seed_vector.extend(parent_b_seeds[:branch_b_tracker.start_seed_index])
            crossover_subvector_to_use = subvector_a
            crossover_insertion_point = branch_b_tracker.start_seed_index
            offspring_initial_seed_vector.extend(parent_b_seeds[branch_b_tracker.end_seed_index:])

        # --- Perform the actual generation with the crossover-aware SeedManager ---
        # The SeedManager now gets the base parent's full seed vector,
        # but also instructions on where to insert the crossover_subvector.

        final_seed_manager = SeedManager(
            initial_seed_vector=offspring_initial_seed_vector,
            crossover_target_nt=target_non_terminal,
            crossover_subvector=crossover_subvector_to_use,
            crossover_insertion_point=crossover_insertion_point # This isn't directly used by SeedManager anymore,
                                                                 # but it clarifies the intent for debugging.
        )
        
        # Now, perform the generation with this configured seed manager
        # The initial template choice also happens via get_choice, so it's part of the seed_manager's flow.
        return self._generate_prompt_from_seeds(
            seed_vector=(parent_a_seeds if self.child_a_or_b else parent_b_seeds), # The base parent's seeds for initial consumption
            template_strings=template_strings, 
            lookups=lookups, 
            base_negatives=base_negatives, 
            seed_manager=final_seed_manager, # Pass our custom configured seed manager
            crossover_target_nt_for_generation=target_non_terminal,
            crossover_subvector_for_generation=crossover_subvector_to_use
        )


    # Helper method to run a full generation given a SeedManager (for both tracing and final gen)
    def _run_dummy_generation(self, template_strings: list, lookups: dict, base_negatives: list, seed_manager: SeedManager):
        """Runs the full prompt generation process to populate seed_manager.branch_traces."""
        # This resets the resolutions_dict for a clean trace run
        original_resolutions_dict = self.resolutions_dict
        self.resolutions_dict = {} 

        # The very first choice (initial_template) for the prompt is also a 'branch'
        # Let's consider 'template' as the non-terminal for the top-level choice.
        seed_manager.start_branch('template') # Mark the beginning of the entire template generation
        initial_template = seed_manager.get_choice(template_strings)
        seed_manager.end_branch() # Mark the end of the initial template choice itself (not its full expansion)

        base, reflection = self.templateExpand(
            initial_template,
            lookups=lookups,
            reflection="",
            seed_manager=seed_manager
        )
        base, appendices = self.iterateExpansions(base, reflection, lookups, seed_manager)

        # Handle other prompt modifications (parens, negatives) which also consume seeds if options
        if random.random() < self.strip_parens_probability:
            base = re.sub(r"[\(\)+]", "", base, 0)
            base = re.sub(r"\-+\s", " ", base, 0)
            base = re.sub(r"\-+,", ",", base, 0)
            appendices = re.sub(r"[\(\)+]", "", appendices, 0)
            appendices = re.sub(r"\-+\s", " ", appendices, 0)
            appendices = re.sub(r"\-+,", ",", appendices, 0)
        if self.remove_negatives:
            base = re.sub(r"\[[^\]\[]*\]", "", base, 0)
            appendices = re.sub(r"\[[^\]\[]*\]", "", appendices, 0)

        # Cleanup does not consume seeds
        base = self.cleanup(base)
        appendices = self.cleanup(appendices)

        if not (self.remove_negatives or (not base_negatives)):
            seed_manager.start_branch('negative_template') # A special non-terminal for negatives
            # This choice is directly from get_choice, so it's tracked
            appendices = appendices + " " + seed_manager.get_choice(base_negatives)
            seed_manager.end_branch()
        
        # Restore original resolutions_dict state
        self.resolutions_dict = original_resolutions_dict
        
        # Note: We don't return the prompt string here, only use this to populate branch_traces.
        # The actual prompt generation happens in _generate_prompt_from_seeds.


    # This method encapsulates the actual prompt generation
    def _generate_prompt_from_seeds(
        self, 
        seed_vector: list, 
        template_strings: list, 
        lookups: dict, 
        base_negatives: list,
        seed_manager: Optional[SeedManager] = None, # Can be pre-configured for crossover
        crossover_target_nt_for_generation: Optional[str] = None,
        crossover_subvector_for_generation: Optional[list] = None
    ) -> HalvedPromptOutput:
        """
        Generates a prompt using a given seed vector and grammar.
        Handles crossover integration if seed_manager is pre-configured.
        """
        # Ensure resolutions_dict is clean for this specific generation run.
        # This is crucial because `invoke` and `_run_dummy_generation` modify it.
        # We need a fresh dict for each *actual* prompt generation.
        temp_resolutions_dict = {}
        # Ensure that templateExpand and iterateExpansions use this temp dict
        # by making it an instance attribute just for the duration of this call
        original_resolutions_dict = self.resolutions_dict
        self.resolutions_dict = temp_resolutions_dict

        if seed_manager is None:
            # For standard generation (e.g., from original PromptFromLookupTable), or fallback
            seed_manager = SeedManager(seed_vector)
        else:
            # For crossover, the provided seed_manager is already configured
            # We need to make sure its internal `seed_vector_in` matches the "base" parent
            # and its current_seed_index is correctly initialized
            seed_manager.seed_vector_in = seed_vector # The base parent's full seeds
            seed_manager.current_seed_index = 0
            # Also, reset its in_crossover_branch flag and subvector index for the actual run
            seed_manager.in_crossover_branch = False # Will be set to True by start_branch
            seed_manager.crossover_subvector_index = 0
            seed_manager.crossover_target_nt = crossover_target_nt_for_generation
            seed_manager.crossover_subvector = crossover_subvector_for_generation


        # The very first choice (initial_template) for the prompt is also a 'branch'
        seed_manager.start_branch('template')
        initial_template = seed_manager.get_choice(template_strings)
        seed_manager.end_branch()

        base, reflection = self.templateExpand(
            initial_template,
            lookups=lookups,
            reflection="",
            seed_manager=seed_manager
        )
        base, appendices = self.iterateExpansions(base, reflection, lookups, seed_manager)

        if random.random() < self.strip_parens_probability:
            base = re.sub(r"[\(\)+]", "", base, 0)
            base = re.sub(r"\-+\s", " ", base, 0)
            base = re.sub(r"\-+,", ",", base, 0)
            appendices = re.sub(r"[\(\)+]", "", appendices, 0)
            appendices = re.sub(r"\-+\s", " ", appendices, 0)
            appendices = re.sub(r"\-+,", ",", appendices, 0)
        if self.remove_negatives:
            base = re.sub(r"\[[^\]\[]*\]", "", base, 0)
            appendices = re.sub(r"\[[^\]\[]*\]", "", appendices, 0)

        base = self.cleanup(base)
        appendices = self.cleanup(appendices)

        if not (self.remove_negatives or (not base_negatives)):
            seed_manager.start_branch('negative_template') # A special non-terminal for negatives
            appendices = appendices + " " + seed_manager.get_choice(base_negatives)
            seed_manager.end_branch() # End the negative template branch

        result = (base + appendices).strip()

        # Restore original resolutions_dict state
        self.resolutions_dict = original_resolutions_dict

        return HalvedPromptOutput(
            prompt=result,
            part_a=base.strip(),
            part_b=appendices.strip(),
            resolutions=json.dumps(temp_resolutions_dict), # Use the temp dict that was populated
            seed_vector=json.dumps(seed_manager.seed_vector_out)
        )

@invocation(
    "generate_evolutionary_prompts",
    title="Generate Evolutionary Prompts",
    tags=["prompt", "lookups", "grammar", "evolution", "GA", "population"],
    category="prompt",
    version="1.5.0",
    use_cache=False,
)
class GenerateEvolutionaryPromptsInvocation(_BasePromptGenerator, BaseInvocation):
    """
    Generates a new population of prompts by performing crossover operations
    on an input list of parent seed vectors.
    """

    # --- Standard fields from PromptFromLookupTableInvocation ---
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

    seed_vectors_in: list[str] = InputField(
        description="List of JSON array strings, each representing a parent's seed vector.",
        default=[]
    )
    target_population_size: int = InputField(
        default=10, ge=1, description="The desired size of the new population to generate."
    )
    selected_pair: int = InputField(
        default=0, ge=0, description="The selected population member to output specifically."
    )
    ga_seed: int = InputField(
        default=0, description="Seed for the random number generator to ensure deterministic GA operations"
    )
    # Optional input to guide crossover, similar to the single crossover node
    crossover_non_terminal: Optional[str] = InputField(
        default=None,
        description="Optional: The non-terminal (key in lookups) to target for the crossover branch. If None, a random one will be chosen from available common non-terminals for each crossover.",
    )


    @validator("lookups")
    def validate_lookups(cls, v):
        valid = False
        if isinstance(v, list):
            for i in v:
                loaded = json.loads(i)
                if ("templates" in loaded) or ("template" in loaded):
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

    def invoke(self, context: InvocationContext) -> EvolutionaryPromptListOutput:
        random.seed(self.ga_seed)
        # Initialize resolutions_dict for this instance (from _BasePromptGenerator)
        self.resolutions_dict = {}

        if not (0 <= self.selected_pair < self.target_population_size):
            raise ValueError("selected_pair must be within the range [0, target_population_size)")
        pair_index = self.selected_pair

        # --- Load and prepare lookups and resolutions ---
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
        self.resolutions_dict = resolutions_dict # Store in instance

        lookups = {}
        if isinstance(self.lookups, list):
            # Parse and store the lookup tables with a sortable key
            parsed_lookups = []
            for lookup_table_str in self.lookups:
                parsed_table = json.loads(lookup_table_str)
                # Create a sortable representation: tuple of sorted (key, value) pairs
                # This makes the sorting deterministic based on content
                sort_key = tuple(sorted(parsed_table.items()))
                parsed_lookups.append((sort_key, parsed_table))

            # Sort the parsed lookup tables deterministically
            parsed_lookups.sort(key=lambda x: x[0])

            for sort_key, lookup_table in reversed(parsed_lookups):
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
        
        # --- Parse all incoming parent seed vectors ---
        parsed_parent_seed_vectors: List[List[int]] = [json.loads(s) for s in self.seed_vectors_in]

        if not parsed_parent_seed_vectors:
            _logger.warning("No parent seed vectors provided. Returning empty population.")
            return EvolutionaryPromptListOutput(prompt_seed_pairs=[], selected_pair="")

        new_population_seed_vectors: List[List[int]] = []
        prompt_seed_pairs_output: List[str] = []

        # We need to trace all potential parents to know their branch structures
        # This can be computationally heavy if the initial population is huge or grammars are deep.
        # Consider caching these traces if this becomes a bottleneck.
        parent_traces: Dict[Tuple[int, ...], List[SeedManager.SeedBranchTracker]] = {}
        all_possible_common_nts = set()

        _logger.info(f"Tracing {len(parsed_parent_seed_vectors)} parent seed vectors...")
        for i, p_seed_vec in enumerate(parsed_parent_seed_vectors):
            temp_seed_manager = SeedManager(p_seed_vec)
            self._run_dummy_generation(template_strings, lookups, base_negatives, temp_seed_manager)
            parent_traces[tuple(p_seed_vec)] = temp_seed_manager.branch_traces
            
            valid_nts_for_parent = {t.non_terminal_name for t in temp_seed_manager.branch_traces if t.end_seed_index is not None}
            # Add 'template' if any generation happened for this parent
            if temp_seed_manager.seed_vector_out:
                valid_nts_for_parent.add('template')
            
            if not all_possible_common_nts:
                all_possible_common_nts = valid_nts_for_parent
            else:
                all_possible_common_nts.intersection_update(valid_nts_for_parent)
        _logger.info(f"Tracing complete. Found common non-terminals: {list(all_possible_common_nts)}")


        # --- Generate new population via crossover ---
        while len(new_population_seed_vectors) < self.target_population_size:
            if len(parsed_parent_seed_vectors) < 2:
                _logger.warning("Not enough parent seed vectors for crossover. Skipping further crossover.")
                # If we have less than 2 parents, we can't do crossover.
                # Just add remaining parents to population or break if population is already full enough.
                if len(new_population_seed_vectors) < len(parsed_parent_seed_vectors):
                     new_population_seed_vectors.extend(parsed_parent_seed_vectors[len(new_population_seed_vectors):])
                break # Exit loop if no more parents or target reached

            # Select two parents randomly (with replacement)
            parent_a_seeds = random.choice(parsed_parent_seed_vectors)
            parent_b_seeds = random.choice(parsed_parent_seed_vectors)

            # Ensure parents are distinct if possible, though not strictly required for GA (can lead to identical children)
            if parent_a_seeds == parent_b_seeds:
                continue # Skip if both parents are identical, to promote diversity

            parent_a_trace = parent_traces[tuple(parent_a_seeds)]
            parent_b_trace = parent_traces[tuple(parent_b_seeds)]

            # Identify common non-terminals for this specific pair of parents
            current_pair_common_nts = (
                {t.non_terminal_name for t in parent_a_trace if t.end_seed_index is not None}
                .intersection(
                    {t.non_terminal_name for t in parent_b_trace if t.end_seed_index is not None}
                )
            )
            if any(t.seed_vector_out for t in [SeedManager(parent_a_seeds), SeedManager(parent_b_seeds)]):
                 current_pair_common_nts.add('template') # Add 'template' as a potential common point

            # Filter common_nts by the user-specified crossover_non_terminal if provided
            eligible_nts_for_crossover = []
            if self.crossover_non_terminal and self.crossover_non_terminal in current_pair_common_nts:
                eligible_nts_for_crossover = [self.crossover_non_terminal]
            else:
                eligible_nts_for_crossover = list(current_pair_common_nts)
            
            if not eligible_nts_for_crossover:
                _logger.debug("No common non-terminals for crossover between selected parents. Skipping this pair.")
                continue # Try another pair of parents

            # Choose the crossover non-terminal for this pair
            target_non_terminal = random.choice(eligible_nts_for_crossover)
            
            # Find the specific branch subvectors for the chosen non-terminal.
            # For simplicity, we'll pick the *first* occurrence of the target non-terminal in each trace.
            branch_a_tracker: Optional[SeedManager.SeedBranchTracker] = None
            for t in parent_a_trace:
                if t.non_terminal_name == target_non_terminal and t.end_seed_index is not None:
                    branch_a_tracker = t
                    break
            
            branch_b_tracker: Optional[SeedManager.SeedBranchTracker] = None
            for t in parent_b_trace:
                if t.non_terminal_name == target_non_terminal and t.end_seed_index is not None:
                    branch_b_tracker = t
                    break
            
            if not (branch_a_tracker and branch_b_tracker):
                 _logger.debug(f"Could not find valid branches for '{target_non_terminal}' in both selected parents. Skipping this pair.")
                 continue # Try another pair of parents


            # Extract the subvectors
            subvector_a = parent_a_seeds[branch_a_tracker.start_seed_index : branch_a_tracker.end_seed_index]
            subvector_b = parent_b_seeds[branch_b_tracker.start_seed_index : branch_b_tracker.end_seed_index]

            _logger.debug(f"Crossover on '{target_non_terminal}'. P_A subvector length: {len(subvector_a)}, P_B subvector length: {len(subvector_b)}")

            # --- Generate Child A (Parent A + B's branch) ---
            child_a_initial_seed_vector = list(parent_a_seeds[:branch_a_tracker.start_seed_index]) # Copy prefix
            child_a_initial_seed_vector.extend(subvector_b) # Insert B's subvector
            child_a_initial_seed_vector.extend(parent_a_seeds[branch_a_tracker.end_seed_index:]) # Copy suffix

            # --- Generate Child B (Parent B + A's branch) ---
            child_b_initial_seed_vector = list(parent_b_seeds[:branch_b_tracker.start_seed_index]) # Copy prefix
            child_b_initial_seed_vector.extend(subvector_a) # Insert A's subvector
            child_b_initial_seed_vector.extend(parent_b_seeds[branch_b_tracker.end_seed_index:]) # Copy suffix

            new_population_seed_vectors.append(child_a_initial_seed_vector)
            if len(new_population_seed_vectors) < self.target_population_size:
                new_population_seed_vectors.append(child_b_initial_seed_vector)
            
            _logger.debug(f"Generated {len(new_population_seed_vectors)}/{self.target_population_size} children.")

        # --- Generate prompts for the new population ---
        _logger.info(f"Generating prompts for new population of size {len(new_population_seed_vectors)}...")
        for i, child_seed_vector in enumerate(new_population_seed_vectors):
            _logger.debug(f"Generating prompt for child {i+1} with seed vector: {child_seed_vector}")
            # Reset self.resolutions_dict for each new prompt generation
            self.resolutions_dict = {} 

            # Create a fresh SeedManager for each child's actual generation
            # This SeedManager is NOT pre-configured for crossover as the crossover has already
            # been "baked into" the child_seed_vector.
            current_seed_manager = SeedManager(initial_seed_vector=child_seed_vector)

            # Re-use the _generate_prompt_from_seeds helper
            generated_output = self._generate_prompt_from_seeds(
                seed_vector=child_seed_vector, # This will be the `initial_seed_vector` for the SeedManager
                template_strings=template_strings,
                lookups=lookups,
                base_negatives=base_negatives,
                seed_manager=current_seed_manager # Pass the freshly configured SeedManager
            )
            
            # Zip prompt and its generated seed vector (which might be longer/shorter than child_seed_vector
            # if random choices were made during generation due to seed vector exhaustion)
            zipped_pair = [generated_output.prompt, generated_output.seed_vector]
            prompt_seed_pairs_output.append(json.dumps(zipped_pair))

        _logger.info(f"Successfully generated {len(prompt_seed_pairs_output)} evolutionary prompts.")
        return EvolutionaryPromptListOutput(prompt_seed_pairs=prompt_seed_pairs_output, selected_pair=prompt_seed_pairs_output[pair_index])

    def _run_dummy_generation(self, template_strings: list, lookups: dict, base_negatives: list, seed_manager: SeedManager):
        """Runs the full prompt generation process to populate seed_manager.branch_traces."""
        # This is essentially a copy of the generate logic, but its purpose is only to populate traces.
        # It needs to set/reset resolutions_dict to ensure a clean trace.
        original_resolutions_dict = self.resolutions_dict
        self.resolutions_dict = {} 

        # The very first choice (initial_template) for the prompt is also a 'branch'
        seed_manager.start_branch('template') 
        initial_template = seed_manager.get_choice(template_strings)
        seed_manager.end_branch() 

        base, reflection = self.templateExpand(
            initial_template,
            lookups=lookups,
            reflection="",
            seed_manager=seed_manager
        )
        base, appendices = self.iterateExpansions(base, reflection, lookups, seed_manager)

        # The following logic for strip_parens_probability, remove_negatives, and base_negatives
        # also consumes seeds if they involve choices, so it needs to be included in the trace run.
        if random.random() < self.strip_parens_probability:
            # These ops don't inherently consume seeds but ensure consistency if the random check passes
            pass 
        if self.remove_negatives:
            pass

        # Cleanup does not consume seeds
        # base = self.cleanup(base) # Don't need to actually clean up for dummy run
        # appendices = self.cleanup(appendices)

        if not (self.remove_negatives or (not base_negatives)):
            seed_manager.start_branch('negative_template') 
            seed_manager.get_choice(base_negatives) # Just consume the seed
            seed_manager.end_branch()
        
        # Restore original resolutions_dict state
        self.resolutions_dict = original_resolutions_dict
        
    def _generate_prompt_from_seeds(
        self, 
        seed_vector: list, 
        template_strings: list, 
        lookups: dict, 
        base_negatives: list,
        seed_manager: Optional[SeedManager] = None, # Can be pre-configured for crossover
        # The following args are specifically for when the seed_manager is doing the crossover during generation
        # but in this `evolutionary` node, crossover is done *before* generation, so these are not strictly used
        # by the SeedManager itself in the final generation phase. Keeping for API consistency if desired.
        crossover_target_nt_for_generation: Optional[str] = None, 
        crossover_subvector_for_generation: Optional[list] = None
    ) -> HalvedPromptOutput:
        """
        Generates a prompt using a given seed vector and grammar.
        Handles crossover integration if seed_manager is pre-configured.
        """
        # Ensure resolutions_dict is clean for this specific generation run.
        temp_resolutions_dict = {}
        original_resolutions_dict = self.resolutions_dict
        self.resolutions_dict = temp_resolutions_dict

        if seed_manager is None:
            seed_manager = SeedManager(seed_vector)
        else:
            # If a seed_manager is provided (e.g., from invoke()), reset its state for THIS generation.
            # Crucially, for this evolutionary node, the `seed_vector` *already contains* the crossover.
            # So, the seed_manager itself should NOT be told to perform another crossover during this final step.
            seed_manager.seed_vector_in = seed_vector # The base parent's full seeds
            seed_manager.current_seed_index = 0
            # Ensure crossover flags are OFF for the *final* generation, as the seeds are pre-mixed.
            seed_manager.in_crossover_branch = False
            seed_manager.crossover_subvector_index = 0
            seed_manager.crossover_target_nt = None # No further crossover at this stage
            seed_manager.crossover_subvector = None # No further crossover at this stage
            seed_manager.seed_vector_out = [] # Clear output seeds for this run

        seed_manager.start_branch('template')
        initial_template = seed_manager.get_choice(template_strings)
        seed_manager.end_branch()

        base, reflection = self.templateExpand(
            initial_template,
            lookups=lookups,
            reflection="",
            seed_manager=seed_manager
        )
        base, appendices = self.iterateExpansions(base, reflection, lookups, seed_manager)

        # These operations should consume seeds if they involve probabilistic choices
        if random.random() < self.strip_parens_probability:
            base = re.sub(r"[\(\)+]", "", base, 0)
            base = re.sub(r"\-+\s", " ", base, 0)
            base = re.sub(r"\-+,", ",", base, 0)
            appendices = re.sub(r"[\(\)+]", "", appendices, 0)
            appendices = re.sub(r"\-+\s", " ", appendices, 0)
            appendices = re.sub(r"\-+,", ",", appendices, 0)
        if self.remove_negatives:
            base = re.sub(r"\[[^\]\[]*\]", "", base, 0)
            appendices = re.sub(r"\[[^\]\[]*\]", "", appendices, 0)

        base = self.cleanup(base)
        appendices = self.cleanup(appendices)

        if not (self.remove_negatives or (not base_negatives)):
            seed_manager.start_branch('negative_template')
            appendices = appendices + " " + seed_manager.get_choice(base_negatives)
            seed_manager.end_branch()

        result = (base + appendices).strip()

        # Restore original resolutions_dict state
        self.resolutions_dict = original_resolutions_dict

        return HalvedPromptOutput(
            prompt=result,
            part_a=base.strip(),
            part_b=appendices.strip(),
            resolutions=json.dumps(temp_resolutions_dict),
            seed_vector=json.dumps(seed_manager.seed_vector_out)
        )

@invocation_output("json_list_strings_output")
class JsonListStringsOutput(BaseInvocationOutput):
    """Output class for two strings extracted from a JSON list."""

    prompt: str = OutputField(description="The prompt string from the JSON list")
    seed_vector: str = OutputField(description="The seed vector string from the JSON list")

@invocation(
    "separate_prompt_and_seed_vector",
    title="Separate Prompt and Seed Vector",
    tags=["json", "list", "split", "string", "prompt", "genetic"],
    category="prompt",
    version="1.5.0",
)
class SeparatePromptAndSeedVectorInvocation(BaseInvocation):
    """
    Parses a JSON string representing a list of two strings,
    outputting each string separately.
    """

    pair_input: str = InputField(
        default='["", ""]',
        description="JSON string of a list containing exactly two strings, e.g., '[\"string one\", \"string two\"]'",
        ui_component=UIComponent.Textarea,
    )

    def invoke(self, context: InvocationContext) -> JsonListStringsOutput:
        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        try:
            parsed_list = json.loads(self.pair_input)

            if not isinstance(parsed_list, list):
                raise ValueError("Input JSON is not a list.")

            if len(parsed_list) != 2:
                raise ValueError("Input JSON list does not contain exactly two elements.")

            string1 = str(parsed_list[0])
            string2 = str(parsed_list[1])

            return JsonListStringsOutput(prompt=string1, seed_vector=string2)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            raise ValueError(f"Invalid JSON input: {e}")
        except ValueError as e:
            logger.error(f"Error processing JSON list: {e}")
            raise ValueError(f"Error processing JSON list: {e}")
