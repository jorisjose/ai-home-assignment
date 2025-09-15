def test_imports():
    import src.config as c
    import src.data_prep as dp
    import src.gcp_nlp as nlp
    import src.vertex_summarize as vs
    import src.agent.workflow as wf
    assert c.SETTINGS is not None
