from src.pipelines.all import run_all
from src.pipelines.ctr import run_ctr_track
from src.pipelines.relevance import run_relevance_track
from src.pipelines.uplift import run_uplift_track
from src.pipelines.ope import run_ope_track
PIPELINES={"all":run_all,"ctr":run_ctr_track,"relevance":run_relevance_track,"uplift":run_uplift_track,"ope":run_ope_track}
