import random
from typing import List, Optional

from vidore_generation.generators.structs import (
    Answerability,
    Modality,
    QueryFormat,
    QueryModule,
    QueryType,
)

FORBIDDEN_FORMATS_PER_TYPE = {
    QueryType.MULTI_HOP: [QueryFormat.KEYWORD],
    QueryType.ENUMERATIVE: [QueryFormat.KEYWORD],
    QueryType.BOOLEAN: [QueryFormat.KEYWORD, QueryFormat.INSTRUCTION],
}


class QueryModuleSampler:
    def __init__(
        self, modules: List[QueryModule], weights: Optional[List[float]] = None
    ) -> None:
        """Initializes the sampler with modules and optional weights."""
        assert len(modules) > 0
        if weights is not None:
            assert len(weights) == len(modules)
        self.modules = modules
        self.weights = weights

    def sample(self, k: int) -> List[QueryModule]:
        """Samples k query modules based on the given seed."""
        modules = []
        for module in self.modules:
            if module.type == QueryType.ANY:
                new_type = random.choice(
                    [x for x in list(QueryType) if x != QueryType.ANY]
                )
            else:
                new_type = module.type
            if module.format == QueryFormat.ANY:
                if module.type in FORBIDDEN_FORMATS_PER_TYPE:
                    new_format = random.choice(
                        [
                            x
                            for x in list(QueryFormat)
                            if x not in FORBIDDEN_FORMATS_PER_TYPE[module.type]
                            and x != QueryFormat.ANY
                        ]
                    )
                else:
                    new_format = random.choice(
                        [x for x in list(QueryFormat) if x != QueryFormat.ANY]
                    )
            else:
                new_format = module.format
            if module.modality == Modality.ANY:
                new_modality = random.choice(
                    [x for x in list(Modality) if x != Modality.ANY]
                )
            else:
                new_modality = module.modality
            new_module = QueryModule(
                type=new_type,
                format=new_format,
                answerability=module.answerability,
                modality=new_modality,
                instruction=module.instruction,
            )
            modules.append(new_module)
        return random.choices(modules, weights=self.weights, k=k)


TableModuleSampler = QueryModuleSampler(
    modules=[
        QueryModule(
            type=QueryType.EXTRACTIVE,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, choose one "
                "important piece of information from any of these"
                " tables. Then, create a clear and specific query that can be fully answered using only that chosen"
                " piece of information.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.BOOLEAN,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, identify one important statement that can be logically concluded"
                " from the data in one of those tables. This statement should be either true (affirmation) or false"
                " (negation) based on the information provided. Your reasoning may involve multiple steps. Then, create"
                " a fully answerable yes/no query that captures this statement.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.BOOLEAN,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, find a complex statement (affirmative or negative) that:\n"
                "a) Involves multiple logical steps to reach\n"
                "b) Could theoretically be partially deduced from one or multiple tables\n"
                "c) Cannot actually be fully deduced because some necessary information is missing from the document\n"
                "You must generate a partially answerable yes/no query that has this complex statement as its"
                " answer.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.NUMERICAL,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, look for relationships between the data that require"
                " some mathematical calculation to reveal an interesting insight. Then, formulate a specific query"
                " that can be answered using this calculated insight. The query should require more than simply"
                " reading numbers directly from the table(s). If no meaningful numerical querys can be derived,"
                " respond with null."
            ),
        ),
        QueryModule(
            type=QueryType.NUMERICAL,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, look for a piece of interesting information that could be"
                " calculated using data from one or more of these tables. However, this calculation should also"
                " require some additional information that is not present on the document. Based on this, create a"
                " partially answerable query that requires a numerical calculation and cannot be completely"
                " answered because some necessary information is missing from the document.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.COMPARE_CONTRAST,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, find two entities or topics within these tables that are closely related or similar."
                " Using these two entities or topics, create a"
                " fully answerable query that requires comparing and/or contrasting the two entities. The"
                " query should encourage analysis of similarities and/or differences between the chosen entities."
                " If you cannot find suitable entities for comparison, or if the table content doesn't allow for"
                " meaningful compare-contrast querys, return null as the result."
            ),
        ),
        QueryModule(
            type=QueryType.COMPARE_CONTRAST,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, choose one entity or topic mentioned in any of these tables."
                " Think of another entity or topic that is related but not present on the document. Create a query"
                " that:\n"
                "a) Asks to compare and/or contrast the chosen entity with the one not on the document\n"
                "b) Can be partially answered using information from the document, but requires "
                "additional information not"
                " present on the document for a complete answer\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.ENUMERATIVE,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TABLE,
            instruction=(
                "If you see tables, identify entities or topics mentioned in these tables that"
                " have some relation to each other. Determine a specific property that is common among these related"
                " entities or topics. Formulate an enumerative query that:\n"
                "a) Asks to list all examples that possess this specific property\n"
                "b) Optionally, requests details about the specifics of each example. The requested details may be"
                " absent from the document.\n"
                "If you can't create such a query return null instead."
            ),
        ),
    ],
    weights=[1, 0.5, 0.5, 1, 1, 1, 1, 1],
)


FigureModuleSampler = QueryModuleSampler(
    modules=[
        QueryModule(
            type=QueryType.EXTRACTIVE,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, choose one important piece of information from any of these"
                " figures. Then, create a clear and specific query that can be fully answered using only that chosen"
                " piece of information.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.OPEN_ENDED,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, based on the information displayed in the figures, create a"
                " comprehensive, open-ended query that can be fully answered using the data or concepts shown. This"
                " query should be broad in scope or focus on qualitative aspects of the information, rather than"
                " asking for specific numerical values or entities.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.OPEN_ENDED,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, create a broad or open-ended query that is inspired by the information"
                " shown in one of the figures on the document. The query should be partially answerable based on the"
                " figure's content, it may also require additional knowledge or interpretation to fully address.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.BOOLEAN,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, identify one important statement that can be logically concluded from the"
                " data in one of those figures. This statement should be either true (affirmation) or false (negation)"
                " based on the information provided. Your reasoning may involve multiple steps. Then, create a yes/no"
                " query that captures this statement.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.BOOLEAN,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, find a complex statement (affirmative or negative) that:\n"
                "a) Involves multiple logical steps to reach\n"
                "b) Could theoretically be partially deduced from one or multiple figures\n"
                "c) Cannot actually be fully deduced because some necessary information is missing from the document\n"
                "You must generate a fully answerable yes/no query that has this complex statement as its answer.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.COMPARE_CONTRAST,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, identify two entities or topics that are closely related to each other."
                " These can be from a single figure or across multiple figures. Based on these two entities or topics,"
                " create a query that requires comparing and/or contrasting them. Ensure that the query can be fully"
                " answered using the information provided in the figure(s). If you cannot find two closely related"
                " entities or topics suitable for a compare-contrast query, or if this task is not applicable to the"
                " content of the document, return null.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.COMPARE_CONTRAST,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, select one entity or topic depicted in any of these figures."
                " Think of another related entity or topic that is not shown in the document. Create a query that:\n"
                "a) Requires comparing and/or contrasting the selected entity with the off-document entity.\n"
                "b) Can be partially answered using information from the document,"
                " but needs additional information to be"
                " fully answered.\n"
                "If it's not possible to create such a compare-contrast query, or if this task doesn't apply to the"
                " content of the document, return null.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.ENUMERATIVE,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.FIGURE,
            instruction=(
                "If you see images or charts, identify entities or topics mentioned in these figures that"
                " have some relation to each other. Determine a specific property that is common among these related"
                " entities or topics. Formulate an enumerative query that:\n"
                "a) Asks to list all examples that possess this specific property\n"
                "b) Optionally, requests details about the specifics of each example. The requested details may be"
                " absent from the document.\n"
                "If you can't create such a query return null instead."
            ),
        ),
    ],
    weights=[2, 3, 3, 1, 1, 1, 1, 1],
)


TextModuleSampler = QueryModuleSampler(
    modules=[
        QueryModule(
            type=QueryType.EXTRACTIVE,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, select one important piece of information from any paragraph. Then, create a"
                " query that can be fully answered using only that key information.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.OPEN_ENDED,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, create a comprehensive, open-ended query that can be fully answered using the"
                " data or concepts shown. This query should be broad in scope or focus on qualitative aspects of the"
                " information, rather than asking for specific numerical values or entities.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.OPEN_ENDED,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, create a broad or open-ended query that is inspired by the information shown in"
                " one of the paragraphs. The query should be partially answerable based on the paragraph's content, it"
                " may also require additional knowledge or interpretation to fully address.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.BOOLEAN,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, identify one important statement that can be logically concluded from the data"
                " in one of those paragraphs. This statement should be either true (affirmation) or false (negation)"
                " based on the information provided. Your reasoning may involve multiple steps. Then, create a yes/no"
                " query that captures this statement.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.BOOLEAN,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, find a complex statement (affirmative or negative) that:\n"
                "a) Involves multiple logical steps to reach\n"
                "b) Could theoretically be partially deduced from information present on the document\n"
                "c) Cannot actually be fully deduced because some necessary information is missing from the document\n"
                "You must generate a fully answerable yes/no query that has this complex statement as its answer.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.COMPARE_CONTRAST,
            format=QueryFormat.ANY,
            answerability=Answerability.FULL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, identify two closely related entities or topics within the text. Based on these"
                " entities or topics, create a query that:\n"
                "a) Requires comparing and/or contrasting them\n"
                "b) Can be fully answered using the information provided in the text\n"
                "If no suitable compare-contrast query can be formulated, or if this task is not applicable to the given"
                " document, return null.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.COMPARE_CONTRAST,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, select one entity or topic mentioned in any of these paragraphs. Think of another"
                " entity or topic that is not mentioned anywhere on the document, but could be meaningfully compared or"
                " contrasted with the selected entity/topic. Formulate a query that:\n"
                "a) Asks to compare and/or contrast the selected entity/topic with the one you thought of"
                "b) Can be partially answered using information from the document, but requires additional knowledge to"
                " fully answer.\n"
                "If you cannot create a relevant compare-contrast query, or if this task doesn't apply to the content of"
                " the document, return null.\n"
                "If you can't create such a query return null instead."
            ),
        ),
        QueryModule(
            type=QueryType.ENUMERATIVE,
            format=QueryFormat.ANY,
            answerability=Answerability.PARTIAL,
            modality=Modality.TEXT,
            instruction=(
                "If you see paragraphs, identify entities or topics mentioned in these paragraphs that have some relation"
                " to each other. Determine a specific property that is common among these related entities or topics."
                " Formulate an enumerative query that:\n"
                "a) Asks to list all examples that possess this specific property\n"
                "b) Optionally, requests details about the specifics of each example. The requested details may be"
                " absent from the document.\n"
                "If you can't create such a query return null instead."
            ),
        ),
    ],
    weights=[2, 3, 3, 1, 1, 1, 1, 1],
)
