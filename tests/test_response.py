import json
from typing import Any, List, Union, cast
from typing_extensions import Annotated

import httpx2
import pytest
import pydantic

from openai import OpenAI, BaseModel, AsyncOpenAI, APIResponseValidationError
from openai._response import (
    APIResponse,
    BaseAPIResponse,
    AsyncAPIResponse,
    BinaryAPIResponse,
    AsyncBinaryAPIResponse,
    extract_response_type,
)
from openai._streaming import Stream
from openai._base_client import FinalRequestOptions

from .utils import rich_print_str


class ConcreteBaseAPIResponse(APIResponse[bytes]): ...


class ConcreteAPIResponse(APIResponse[List[str]]): ...


class ConcreteAsyncAPIResponse(APIResponse[httpx2.Response]): ...


def test_extract_response_type_direct_classes() -> None:
    assert extract_response_type(BaseAPIResponse[str]) == str
    assert extract_response_type(APIResponse[str]) == str
    assert extract_response_type(AsyncAPIResponse[str]) == str


def test_extract_response_type_direct_class_missing_type_arg() -> None:
    with pytest.raises(
        RuntimeError,
        match="Expected type <class 'openai._response.AsyncAPIResponse'> to have a type argument at index 0 but it did not",
    ):
        extract_response_type(AsyncAPIResponse)


def test_extract_response_type_concrete_subclasses() -> None:
    assert extract_response_type(ConcreteBaseAPIResponse) == bytes
    assert extract_response_type(ConcreteAPIResponse) == List[str]
    assert extract_response_type(ConcreteAsyncAPIResponse) == httpx2.Response


def test_extract_response_type_binary_response() -> None:
    assert extract_response_type(BinaryAPIResponse) == bytes
    assert extract_response_type(AsyncBinaryAPIResponse) == bytes


class PydanticModel(pydantic.BaseModel): ...


def test_response_parse_mismatched_basemodel(client: OpenAI) -> None:
    response = APIResponse(
        raw=httpx2.Response(200, content=b"foo"),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    with pytest.raises(
        TypeError,
        match="Pydantic models must subclass our base model type, e.g. `from openai import BaseModel`",
    ):
        response.parse(to=PydanticModel)


@pytest.mark.asyncio
async def test_async_response_parse_mismatched_basemodel(async_client: AsyncOpenAI) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(200, content=b"foo"),
        client=async_client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    with pytest.raises(
        TypeError,
        match="Pydantic models must subclass our base model type, e.g. `from openai import BaseModel`",
    ):
        await response.parse(to=PydanticModel)


def test_response_parse_custom_stream(client: OpenAI) -> None:
    response = APIResponse(
        raw=httpx2.Response(200, content=b"foo"),
        client=client,
        stream=True,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    stream = response.parse(to=Stream[int])
    assert stream._cast_to == int


@pytest.mark.asyncio
async def test_async_response_parse_custom_stream(async_client: AsyncOpenAI) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(200, content=b"foo"),
        client=async_client,
        stream=True,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    stream = await response.parse(to=Stream[int])
    assert stream._cast_to == int


class CustomModel(BaseModel):
    foo: str
    bar: int


def test_response_parse_custom_model(client: OpenAI) -> None:
    response = APIResponse(
        raw=httpx2.Response(200, content=json.dumps({"foo": "hello!", "bar": 2})),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = response.parse(to=CustomModel)
    assert obj.foo == "hello!"
    assert obj.bar == 2


@pytest.mark.asyncio
async def test_async_response_parse_custom_model(async_client: AsyncOpenAI) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(200, content=json.dumps({"foo": "hello!", "bar": 2})),
        client=async_client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = await response.parse(to=CustomModel)
    assert obj.foo == "hello!"
    assert obj.bar == 2


def test_response_basemodel_request_id(client: OpenAI) -> None:
    response = APIResponse(
        raw=httpx2.Response(
            200,
            headers={"x-request-id": "my-req-id"},
            content=json.dumps({"foo": "hello!", "bar": 2}),
        ),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = response.parse(to=CustomModel)
    assert obj._request_id == "my-req-id"
    assert obj.foo == "hello!"
    assert obj.bar == 2
    assert obj.to_dict() == {"foo": "hello!", "bar": 2}
    assert "_request_id" not in rich_print_str(obj)
    assert "__exclude_fields__" not in rich_print_str(obj)


@pytest.mark.asyncio
async def test_async_response_basemodel_request_id(client: OpenAI) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(
            200,
            headers={"x-request-id": "my-req-id"},
            content=json.dumps({"foo": "hello!", "bar": 2}),
        ),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = await response.parse(to=CustomModel)
    assert obj._request_id == "my-req-id"
    assert obj.foo == "hello!"
    assert obj.bar == 2
    assert obj.to_dict() == {"foo": "hello!", "bar": 2}


def test_response_parse_annotated_type(client: OpenAI) -> None:
    response = APIResponse(
        raw=httpx2.Response(200, content=json.dumps({"foo": "hello!", "bar": 2})),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = response.parse(
        to=cast("type[CustomModel]", Annotated[CustomModel, "random metadata"]),
    )
    assert obj.foo == "hello!"
    assert obj.bar == 2


async def test_async_response_parse_annotated_type(async_client: AsyncOpenAI) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(200, content=json.dumps({"foo": "hello!", "bar": 2})),
        client=async_client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = await response.parse(
        to=cast("type[CustomModel]", Annotated[CustomModel, "random metadata"]),
    )
    assert obj.foo == "hello!"
    assert obj.bar == 2


@pytest.mark.parametrize(
    "content, expected",
    [
        ("false", False),
        ("true", True),
        ("False", False),
        ("True", True),
        ("TrUe", True),
        ("FalSe", False),
        (" \ttrue\r\n", True),
        (" \tfalse\r\n", False),
    ],
)
def test_response_parse_bool(client: OpenAI, content: str, expected: bool) -> None:
    response = APIResponse(
        raw=httpx2.Response(200, content=content),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    result = response.parse(to=bool)
    assert result is expected


@pytest.mark.parametrize(
    "content, expected",
    [
        ("false", False),
        ("true", True),
        ("False", False),
        ("True", True),
        ("TrUe", True),
        ("FalSe", False),
        (" \ttrue\r\n", True),
        (" \tfalse\r\n", False),
    ],
)
async def test_async_response_parse_bool(client: AsyncOpenAI, content: str, expected: bool) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(200, content=content),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    result = await response.parse(to=bool)
    assert result is expected


class OtherModel(BaseModel):
    a: str


@pytest.mark.parametrize("client", [False], indirect=True)  # loose validation
def test_response_parse_expect_model_union_non_json_content(client: OpenAI) -> None:
    response = APIResponse(
        raw=httpx2.Response(200, content=b"foo", headers={"Content-Type": "application/text"}),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = response.parse(to=cast(Any, Union[CustomModel, OtherModel]))
    assert isinstance(obj, str)
    assert obj == "foo"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False], indirect=True)  # loose validation
async def test_async_response_parse_expect_model_union_non_json_content(async_client: AsyncOpenAI) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(200, content=b"foo", headers={"Content-Type": "application/text"}),
        client=async_client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = await response.parse(to=cast(Any, Union[CustomModel, OtherModel]))
    assert isinstance(obj, str)
    assert obj == "foo"


JSON_CONTENT_TYPES = [
    "application/json",
    "Application/JSON",
    "application/JSON",
    "APPLICATION/JSON; CHARSET=UTF-8",
    "application/json ; charset=utf-8",
    "application/json\t; charset=utf-8",
    "  application/json  ; charset=utf-8",
    "application/json   ; charset=utf-8",
    "application/json ; charset=utf-8 ; version=1",
    "application/problem+json",
    "application/problem+json ; charset=utf-8",
    "application/vnd.api+json",
    "application/vnd.api+json ; charset=utf-8",
    "APPLICATION/PROBLEM+JSON ; CHARSET=UTF-8",
]


@pytest.mark.parametrize("content_type", JSON_CONTENT_TYPES)
def test_response_parse_content_type_normalization_dict(client: OpenAI, content_type: str) -> None:
    response = APIResponse(
        raw=httpx2.Response(
            200,
            content=json.dumps({"message": "success", "count": 42}),
            headers={"Content-Type": content_type},
        ),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=dict,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    result = response.parse()
    assert isinstance(result, dict)
    assert result == {"message": "success", "count": 42}


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", JSON_CONTENT_TYPES)
async def test_async_response_parse_content_type_normalization_dict(
    async_client: AsyncOpenAI, content_type: str
) -> None:
    response = AsyncAPIResponse(
        raw=httpx2.Response(
            200,
            content=json.dumps({"message": "success", "count": 42}),
            headers={"Content-Type": content_type},
        ),
        client=async_client,
        stream=False,
        stream_cls=None,
        cast_to=dict,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    result = await response.parse()
    assert isinstance(result, dict)
    assert result == {"message": "success", "count": 42}


@pytest.mark.parametrize("content_type", JSON_CONTENT_TYPES)
def test_response_parse_content_type_normalization_list(client: OpenAI, content_type: str) -> None:
    response = APIResponse(
        raw=httpx2.Response(
            200,
            content=json.dumps(["item1", "item2"]),
            headers={"Content-Type": content_type},
        ),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=list,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    result = response.parse()
    assert isinstance(result, list)
    assert result == ["item1", "item2"]


@pytest.mark.parametrize("content_type", JSON_CONTENT_TYPES)
def test_response_parse_content_type_normalization_custom_model(client: OpenAI, content_type: str) -> None:
    response = APIResponse(
        raw=httpx2.Response(
            200,
            content=json.dumps({"foo": "hello!", "bar": 2}),
            headers={"Content-Type": content_type},
        ),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=str,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    obj = response.parse(to=CustomModel)
    assert isinstance(obj, CustomModel)
    assert obj.foo == "hello!"
    assert obj.bar == 2


NON_JSON_HEADERS = [
    {"Content-Type": "text/plain"},
    {"Content-Type": "text/plain ; charset=utf-8"},
    {"Content-Type": "application/octet-stream"},
    {"Content-Type": "text/html"},
    {"Content-Type": "application/xml"},
    {"Content-Type": ""},
    {},
]


@pytest.mark.parametrize("headers", NON_JSON_HEADERS)
def test_response_parse_non_json_content_type_strict_error(client: OpenAI, headers: dict[str, str]) -> None:
    response = APIResponse(
        raw=httpx2.Response(
            200,
            content=b"plain text payload",
            headers=headers,
            request=httpx2.Request("GET", "http://localhost/foo"),
        ),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=dict,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    with pytest.raises(APIResponseValidationError):
        response.parse()


@pytest.mark.parametrize("client", [False], indirect=True)  # loose validation
@pytest.mark.parametrize("headers", NON_JSON_HEADERS)
def test_response_parse_non_json_content_type_loose_fallback(client: OpenAI, headers: dict[str, str]) -> None:
    response = APIResponse(
        raw=httpx2.Response(
            200,
            content=b"plain text payload",
            headers=headers,
            request=httpx2.Request("GET", "http://localhost/foo"),
        ),
        client=client,
        stream=False,
        stream_cls=None,
        cast_to=dict,
        options=FinalRequestOptions.construct(method="get", url="/foo"),
    )

    result = response.parse()
    assert isinstance(result, str)
    assert result == "plain text payload"
