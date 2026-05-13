from pathlib import Path
from ic_core.io_xml import load_glyphs

FIXTURE = Path(__file__).parent / "fixtures" / "Interactive_Classifier_GameraXML_TrainingData.xml"

def test_load_training_data_glyph_count():
    glyphs = load_glyphs(FIXTURE)
    assert len(glyphs) > 0
    g = glyphs[0]
    assert "class_name" in g
    assert "id_state_manual" in g
