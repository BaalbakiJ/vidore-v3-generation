import random
from typing import List, Optional

from vidore_generation.query_generation.vidore_juicer.structs import (
    Answerability,
    Modality,
    QuestionFormat,
    QuestionModule,
    QuestionType,
)

FORBIDDEN_FORMATS_PER_TYPE = {
    QuestionType.MULTI_HOP: [QuestionFormat.KEYWORD],
    QuestionType.ENUMERATIVE: [QuestionFormat.KEYWORD],
    QuestionType.BOOLEAN: [QuestionFormat.KEYWORD, QuestionFormat.INSTRUCTION],
}


class QuestionModuleSampler:
    def __init__(
        self, modules: List[QuestionModule], weights: Optional[List[float]] = None
    ) -> None:
        """Initializes the sampler with modules and optional weights."""
        assert len(modules) > 0
        if weights is not None:
            assert len(weights) == len(modules)
        self.modules = modules
        self.weights = weights

    def sample(self, k: int, seed: Optional[int] = None) -> List[QuestionModule]:
        """Samples k question modules based on the given seed."""
        rng = random.Random(seed)
        modules = []
        for module in self.modules:
            if module.type == QuestionType.ANY:
                new_type = rng.choice(
                    [x for x in list(QuestionType) if x != QuestionType.ANY]
                )
            else:
                new_type = module.type
            if module.format == QuestionFormat.ANY:
                if module.type in FORBIDDEN_FORMATS_PER_TYPE:
                    new_format = random.choice(
                        [
                            x
                            for x in list(QuestionFormat)
                            if x not in FORBIDDEN_FORMATS_PER_TYPE[module.type]
                            and x != QuestionFormat.ANY
                        ]
                    )
                else:
                    new_format = random.choice(
                        [x for x in list(QuestionFormat) if x != QuestionFormat.ANY]
                    )
            else:
                new_format = module.format
            if module.modality == Modality.ANY:
                new_modality = rng.choice(
                    [x for x in list(Modality) if x != Modality.ANY]
                )
            else:
                new_modality = module.modality
            new_module = QuestionModule(
                type=new_type,
                format=new_format,
                answerability=module.answerability,
                modality=new_modality,
                instruction=module.instruction,
            )
            modules.append(new_module)
        return rng.choices(modules, weights=self.weights, k=k)


AdversarialModuleSampler = QuestionModuleSampler(
    modules=[
        QuestionModule(
            type=QuestionType.ANY,
            format=QuestionFormat.ANY,
            answerability=Answerability.ADVERSARIAL,
            modality=Modality.ANY,
            instruction=(
                "Create a tricky unanswerable question that appears answerable at first glance, but is actually"
                " impossible to answer correctly based solely on the information provided on this document."
            ),
        ),
    ]
)

TableModuleSampler = QuestionModuleSampler(
    modules=[
        QuestionModule(
            type=QuestionType.EXTRACTIVE,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document and choose one "
                "important piece of information from any of these"
                " tables. Then, create a clear and specific question that can be fully answered using only that chosen"
                " piece of information."
            ),
        ),
        QuestionModule(
            type=QuestionType.BOOLEAN,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document and identify one important statement that can"
                " be logically concluded"
                " from the data in one of those tables. This statement should be either true (affirmation) or false"
                " (negation) based on the information provided. Your reasoning may involve multiple steps. Then, create"
                " a fully answerable yes/no question that captures this statement."
            ),
        ),
        QuestionModule(
            type=QuestionType.BOOLEAN,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document. Find a complex statement "
                "(affirmative or negative) that:\n"
                "a) Involves multiple logical steps to reach\n"
                "b) Could theoretically be partially deduced from one or multiple tables\n"
                "c) Cannot actually be fully deduced because some necessary information is missing from the document\n"
                "You must generate a partially answerable yes/no question that has this complex statement as its"
                " answer."
            ),
        ),
        QuestionModule(
            type=QuestionType.NUMERICAL,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document. Look for relationships between"
                " the data that require"
                " some mathematical calculation to reveal an interesting insight. Then, formulate a specific question"
                " that can be answered using this calculated insight. The question should require more than simply"
                " reading numbers directly from the table(s). If no meaningful numerical questions can be derived,"
                " respond with null."
            ),
        ),
        QuestionModule(
            type=QuestionType.NUMERICAL,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document. Look for a piece of "
                "interesting information that could be"
                " calculated using data from one or more of these tables. However, this calculation should also"
                " require some additional information that is not present on the document. Based on this, create a"
                " partially answerable question that requires a numerical calculation and cannot be completely"
                " answered because some necessary information is missing from the document."
            ),
        ),
        QuestionModule(
            type=QuestionType.COMPARE_CONTRAST,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document. Find two entities or topics within"
                " these tables that are closely related or similar. Using these two entities or topics, create a"
                " fully answerable question that requires comparing and/or contrasting the two entities. The"
                " question should encourage analysis of similarities and/or differences between the chosen entities."
                " If you cannot find suitable entities for comparison, or if the table content doesn't allow for"
                " meaningful compare-contrast questions, return null as the result."
            ),
        ),
        QuestionModule(
            type=QuestionType.COMPARE_CONTRAST,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document. Choose one entity or topic "
                "mentioned in any of these tables."
                " Think of another entity or topic that is related but not present on the document. Create a question"
                " that:\n"
                "a) Asks to compare and/or contrast the chosen entity with the one not on the document\n"
                "b) Can be partially answered using information from the document, but requires "
                "additional information not"
                " present on the document for a complete answer\n"
                "If you can't create such a question return null instead."
            ),
        ),
        QuestionModule(
            type=QuestionType.ENUMERATIVE,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "Imagine tables that are likely to appear in the document. Identify entities or"
                " topics mentioned in these tables that"
                " have some relation to each other. Determine a specific property that is common among these related"
                " entities or topics. Formulate an enumerative question that:\n"
                "a) Asks to list all examples that possess this specific property\n"
                "b) Optionally, requests details about the specifics of each example. The requested details may be"
                " absent from the document."
            ),
        ),
    ],
    weights=[1, 0.5, 0.5, 1, 1, 1, 1, 1],
)


FigureModuleSampler = QuestionModuleSampler(
    modules=[
        QuestionModule(
            type=QuestionType.EXTRACTIVE,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document and choose one "
                "important piece of information from any of these"
                " figures. Then, create a clear and specific question that can be fully answered using only that chosen"
                " piece of information."
            ),
        ),
        QuestionModule(
            type=QuestionType.OPEN_ENDED,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document. Based on the information "
                "displayed in the figures, create a"
                " comprehensive, open-ended question that can be fully answered using the data or concepts shown. This"
                " question should be broad in scope or focus on qualitative aspects of the information, rather than"
                " asking for specific numerical values or entities."
            ),
        ),
        QuestionModule(
            type=QuestionType.OPEN_ENDED,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document Create a broad or "
                "open-ended question that is inspired by"
                " the information shown in one of the figures on the document. The question "
                "should be partially answerable"
                " based on the figure's content, it may also require additional knowledge or interpretation to fully"
                " address."
            ),
        ),
        QuestionModule(
            type=QuestionType.BOOLEAN,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document and identify one important "
                "statement that can be logically concluded"
                " from the data in one of those figures. This statement should be either true (affirmation) or false"
                " (negation) based on the information provided. Your reasoning may involve multiple steps. Then, create"
                " a yes/no question that captures this statement."
            ),
        ),
        QuestionModule(
            type=QuestionType.BOOLEAN,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document. Find a complex statement "
                "(affirmative or negative) that:\n"
                "a) Involves multiple logical steps to reach\n"
                "b) Could theoretically be partially deduced from one or multiple figures\n"
                "c) Cannot actually be fully deduced because some necessary information is missing from the document\n"
                "You must generate a fully answerable yes/no question that has this complex statement as its answer."
            ),
        ),
        QuestionModule(
            type=QuestionType.COMPARE_CONTRAST,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document. Identify two entities or "
                "topics that are closely related to each"
                " other. These can be from a single figure or across multiple figures. Based on these two entities or"
                " topics, create a question that requires comparing and/or contrasting them. Ensure that the question"
                " can be fully answered using the information provided in the figure(s). If you cannot find two closely"
                " related entities or topics suitable for a compare-contrast question, or if this task is not"
                " applicable to the content of the document, return null."
            ),
        ),
        QuestionModule(
            type=QuestionType.COMPARE_CONTRAST,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document. Select one entity or "
                "topic depicted in any of these figures."
                " Think of another related entity or topic that is not shown in the document. Create a question that:\n"
                "a) Requires comparing and/or contrasting the selected entity with the off-document entity.\n"
                "b) Can be partially answered using information from the document,"
                " but needs additional information to be"
                " fully answered.\n"
                "If it's not possible to create such a compare-contrast question, or if this task doesn't apply to the"
                " content of the document, return null."
            ),
        ),
        QuestionModule(
            type=QuestionType.ENUMERATIVE,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "Imagine figures that are likely to appear in the document. Identify entities or "
                "topics mentioned in these figures that"
                " have some relation to each other. Determine a specific property that is common among these related"
                " entities or topics. Formulate an enumerative question that:\n"
                "a) Asks to list all examples that possess this specific property\n"
                "b) Optionally, requests details about the specifics of each example. The requested details may be"
                " absent from the document."
            ),
        ),
    ],
    weights=[2, 3, 3, 1, 1, 1, 1, 1],
)


TextModuleSampler = QuestionModuleSampler(
    modules=[
        QuestionModule(
            type=QuestionType.EXTRACTIVE,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document. Select one important "
                "piece of information from any"
                " paragraph. Then, create a question that can be fully answered using only that key information."
            ),
        ),
        QuestionModule(
            type=QuestionType.OPEN_ENDED,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document. Create a comprehensive, "
                "open-ended question that can be"
                " fully answered using the data or concepts shown. This question should be broad in scope or focus on"
                " qualitative aspects of the information, rather than asking for specific numerical values or entities."
            ),
        ),
        QuestionModule(
            type=QuestionType.OPEN_ENDED,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document. Create a broad or "
                "open-ended question that is inspired"
                " by the information shown in one of the paragraphs. The question should be partially answerable based"
                " on the paragrapgh's content, it may also require additional knowledge or interpretation to fully"
                " address."
            ),
        ),
        QuestionModule(
            type=QuestionType.BOOLEAN,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document and identify "
                "one important statement that can be"
                " logically concluded from the data in one of those paragraphs. This statement should be either true"
                " (affirmation) or false (negation) based on the information provided. Your reasoning may involve"
                " multiple steps. Then, create a yes/no question that captures this statement."
            ),
        ),
        QuestionModule(
            type=QuestionType.BOOLEAN,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document. Find a "
                "complex statement (affirmative or negative)"
                " that:\n"
                "a) Involves multiple logical steps to reach\n"
                "b) Could theoretically be partially deduced from information present on the document\n"
                "c) Cannot actually be fully deduced because some necessary information is missing from the document\n"
                "You must generate a fully answerable yes/no question that has this complex statement as its answer."
            ),
        ),
        QuestionModule(
            type=QuestionType.COMPARE_CONTRAST,
            format=QuestionFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document. "
                "Identify two closely related entities or topics within"
                " the text. Based on these entities or topics, create a question that:\n"
                "a) Requires comparing and/or contrasting them\n"
                "b) Can be fully answered using the information provided in the text\n"
                "If no suitable compare-contrast question can be formulated, or if this task is not applicable to the"
                " given document, return null."
            ),
        ),
        QuestionModule(
            type=QuestionType.COMPARE_CONTRAST,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document. "
                "Select one entity or topic mentioned in any of these"
                " paragraphs. "
                "Think of another entity or topic that is not mentioned anywhere on the document, but could"
                " be meaningfully compared or contrasted with the selected entity/topic. Formulate a question that:\n"
                "a) Asks to compare and/or contrast the selected entity/topic with the one you thought of"
                "b) Can be partially answered using information from the document, but requires additional knowledge to"
                " fully answer\n"
                "If you cannot create a relevant compare-contrast question, or if this task doesn't apply to the"
                " content of the document, return null."
            ),
        ),
        QuestionModule(
            type=QuestionType.ENUMERATIVE,
            format=QuestionFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "Imagine paragraphs that are likely to appear in the document. "
                "Identify entities or topics mentioned in these"
                " paragraphs that have some relation to each other. Determine a specific property that is common among"
                " these related entities or topics. Formulate an enumerative question that:\n"
                "a) Asks to list all examples that possess this specific property\n"
                "b) Optionally, requests details about the specifics of each example. The requested details may be"
                " absent from the document."
            ),
        ),
    ],
    weights=[2, 3, 3, 1, 1, 1, 1, 1],
)
