import pytest

from hillstrom.data import connect

# Six rows in raw-file format, covering all arms, the "Surburban" typo and
# all three purchase-history groups.
SYNTHETIC_CSV = """recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,visit,conversion,spend
10,2) $100 - $200,142.44,1,0,Surburban,0,Phone,Womens E-Mail,0,0,0
6,3) $200 - $350,329.08,1,1,Rural,1,Web,No E-Mail,0,0,0
7,2) $100 - $200,180.65,0,1,Surburban,1,Web,Womens E-Mail,1,1,45.5
9,5) $500 - $750,675.83,1,0,Rural,1,Web,Mens E-Mail,1,0,0
2,1) $0 - $100,45.34,1,0,Urban,0,Web,Womens E-Mail,0,0,0
1,"7) $1,000 +",1050.00,0,1,Urban,0,Multichannel,Mens E-Mail,1,1,120.0
"""


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text(SYNTHETIC_CSV)
    return path


@pytest.fixture
def synthetic_con(synthetic_csv):
    con = connect(synthetic_csv)
    yield con
    con.close()
