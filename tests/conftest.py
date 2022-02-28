import pytest

def pytest_addoption(parser):
    parser.addoption("--resource", default="numpy")

@pytest.fixture(scope="session")
def resource(request):
    import impy as ip
    res = request.config.getoption('--resource')
    return res
