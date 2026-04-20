import sys
from typing import Any, Optional

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    DEFAULT_CONTENT_LAYERS,
    DOCUMENT_TOKENS_EXPORT_LABELS,
    ContentLayer,
)
from docling_core.types.doc.labels import DocItemLabel
from typing_extensions import override


class CustomMarkdownDocSerializer(MarkdownDocSerializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts."""
        text_res = "\n\n".join([p.text for p in parts if p.text])

        if self.requires_page_break():
            page_sep = self.params.page_break_placeholder or ""
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                splits = full_match.split("_")
                start_page_number = int(splits[-4])
                end_page_number = int(splits[-3])
                number_of_pages = end_page_number - start_page_number
                multi_page_sep = "\n\n".join([page_sep] * number_of_pages)
                text_res = text_res.replace(full_match, multi_page_sep)

        return create_ser_result(text=text_res, span_source=parts)


def custom_export_to_markdown(  # noqa: C901
    self,
    delim: str = "\n\n",
    from_element: int = 0,
    to_element: int = sys.maxsize,
    labels: Optional[set[DocItemLabel]] = None,
    strict_text: bool = False,
    escape_underscores: bool = True,
    image_placeholder: str = "<!-- image -->",
    enable_chart_tables: bool = True,
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
    indent: int = 4,
    text_width: int = -1,
    page_no: Optional[int] = None,
    included_content_layers: Optional[set[ContentLayer]] = None,
    page_break_placeholder: Optional[str] = None,  # e.g. "<!-- page break -->",
    include_annotations: bool = True,
    mark_annotations: bool = False,
) -> str:
    my_labels = labels if labels is not None else DOCUMENT_TOKENS_EXPORT_LABELS
    my_layers = (
        included_content_layers
        if included_content_layers is not None
        else DEFAULT_CONTENT_LAYERS
    )
    serializer = CustomMarkdownDocSerializer(
        doc=self,
        params=MarkdownParams(
            labels=my_labels,
            layers=my_layers,
            pages={page_no} if page_no is not None else None,
            start_idx=from_element,
            stop_idx=to_element,
            escape_underscores=escape_underscores,
            image_placeholder=image_placeholder,
            enable_chart_tables=enable_chart_tables,
            image_mode=image_mode,
            indent=indent,
            wrap_width=text_width if text_width > 0 else None,
            page_break_placeholder=page_break_placeholder,
            include_annotations=include_annotations,
            mark_annotations=mark_annotations,
        ),
    )
    ser_res = serializer.serialize()

    # if delim != "\n\n":
    #     _logger.warning(
    #         "Parameter `delim` has been deprecated and will be ignored.",
    #     )
    # if strict_text:
    #     _logger.warning(
    #         "Parameter `strict_text` has been deprecated and will be ignored.",
    #     )

    return ser_res.text
