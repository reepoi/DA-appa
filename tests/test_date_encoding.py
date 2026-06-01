import math
import pytest
import torch
from datetime import datetime, timedelta

from appa.date import format_blanket_date, get_local_time_encoding, get_year_progress_encoding, split_interval, interval_to_tensor


@pytest.mark.parametrize(
    "batch_shape, input_date, expected_encoding",
    [
        (
            (),
            torch.tensor([[2019, 1, 1, 0]]),
            torch.tensor([[math.sin(0), math.cos(0)]]),
        ),
        (
            (2, 3),
            torch.tensor([[2019, 12, 31, 23]]),
            torch.tensor([
                [
                    math.sin(2 * math.pi * ((24 * 364) + 23) / (365 * 24)),
                    math.cos(2 * math.pi * ((24 * 364) + 23) / (365 * 24)),
                ]
            ]),
        ),
        (
            (2,),
            # 366/2=183 -> July 2nd is the 183rd day of the year
            torch.tensor([[2020, 7, 2, 0]]),
            torch.tensor([
                [math.sin(2 * math.pi * (183 / 366)), math.cos(2 * math.pi * (183 / 366))]
            ]),
        ),
        (
            (),
            # 366/4=91.5 -> April 1st is the 91st day of the year, 12:00 is the 0.5th day
            torch.tensor([[2020, 4, 1, 12]]),
            torch.tensor([
                [math.sin(2 * math.pi * (91.5 / 366)), math.cos(2 * math.pi * (91.5 / 366))]
            ]),
        ),
    ],
)
def test_get_year_progress_encoding(batch_shape, input_date, expected_encoding):
    size = (*batch_shape, *input_date.shape)
    batch_input_date = torch.ones(*size) * input_date

    size = (*batch_shape, *expected_encoding.shape)
    batch_exp_encoding = torch.ones(*size) * expected_encoding

    output = get_year_progress_encoding(batch_input_date.long())
    assert torch.allclose(
        output, batch_exp_encoding, atol=1e-6
    ), f"Expected {batch_exp_encoding}, got {output}"


@pytest.mark.parametrize(
    "batch_shape, input_date, input_longitude, expected_encoding",
    [
        (
            (),
            torch.tensor([[2023, 1, 1, 0]]),
            torch.tensor([[0.0]]),
            torch.tensor([[[math.sin(2 * math.pi * 0), math.cos(2 * math.pi * 0)]]]),
        ),
        (
            (2, 3),
            torch.tensor([[2023, 1, 1, 12]]),
            torch.tensor([[180.0]]),
            torch.tensor([
                [
                    [
                        math.sin(2 * math.pi * (((12 + 24 * (180 / 360)) % 24) / 24)),
                        math.cos(2 * math.pi * (((12 + 24 * (180 / 360)) % 24) / 24)),
                    ]
                ]
            ]),
        ),
        (
            (2,),
            torch.tensor([[2023, 1, 1, 0]]),
            torch.tensor([[-180.0]]),
            torch.tensor([
                [
                    [
                        math.sin(2 * math.pi * (((0 + 24 * (180 / 360)) % 24) / 24)),
                        math.cos(2 * math.pi * (((0 + 24 * (180 / 360)) % 24) / 24)),
                    ]
                ]
            ]),
        ),
        (
            (),
            torch.tensor([[2023, 1, 1, 6]]),
            torch.tensor([[90.0]]),
            torch.tensor([
                [
                    [
                        math.sin(2 * math.pi * (((6 + 24 * (90 / 360)) % 24) / 24)),
                        math.cos(2 * math.pi * (((6 + 24 * (90 / 360)) % 24) / 24)),
                    ]
                ]
            ]),
        ),
        (
            (5,),
            torch.tensor([[2023, 1, 1, 18]]),
            torch.tensor([[-90.0]]),
            torch.tensor([
                [
                    [
                        math.sin(2 * math.pi * (((18 + 24 * (270 / 360)) % 24) / 24)),
                        math.cos(2 * math.pi * (((18 + 24 * (270 / 360)) % 24) / 24)),
                    ]
                ]
            ]),
        ),
    ],
)
def test_get_local_time_encoding(batch_shape, input_date, input_longitude, expected_encoding):
    size = (*batch_shape, *input_date.shape)
    batch_input_date = torch.ones(*size) * input_date

    size = (*batch_shape, *expected_encoding.shape)
    batch_exp_encoding = torch.ones(*size) * expected_encoding

    output = get_local_time_encoding(batch_input_date.long(), input_longitude)
    assert torch.allclose(
        output, batch_exp_encoding, atol=1e-6
    ), f"Expected {batch_exp_encoding}, got {output}"


@pytest.mark.parametrize(
    "start_date, end_date, expected_date",
    [
        (
            torch.tensor([[2023, 1, 1, 18]]),
            torch.tensor([[2023, 1, 1, 22]]),
            "01/01/2023 18h to 22h",
        ),
        (
            torch.tensor([[2000, 6, 24, 13]]),
            torch.tensor([[2000, 6, 25, 13]]),
            "24/06/2000 13h to 25/06/2000 13h",
        ),
        (
            torch.tensor([[2010, 2, 5, 4]]),
            torch.tensor([[2010, 3, 5, 4]]),
            "05/02/2010 04h to 05/03/2010 04h",
        ),
        (
            torch.tensor([[2022, 12, 31, 22]]),
            torch.tensor([[2023, 1, 1, 4]]),
            "31/12/2022 22h to 01/01/2023 04h",
        ),
        (
            torch.tensor([[1896, 8, 27, 9]]),
            torch.tensor([[1896, 8, 27, 10]]),
            "27/08/1896 09h to 10h",
        ),
    ],
)
def test_date_to_string(start_date, end_date, expected_date):
    num_dates = torch.randint(0, 10, (1,)).item()
    rand_dates_between = torch.randint(0, 100, (num_dates, 4))
    blanket_dates = torch.cat([start_date, rand_dates_between, end_date])

    date_str = format_blanket_date(blanket_dates)

    assert date_str == expected_date, f"Expected {expected_date}, got {date_str}"


@pytest.mark.parametrize('hours', [1, 6])
def test_interval_to_tensor(hours):
    t = interval_to_tensor('1970-01-01', '1970-01-01', dt=timedelta(hours=hours))
    assert t.shape[0] * hours == 24
