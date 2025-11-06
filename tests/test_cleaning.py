from src.data_prep.cleaning import mask_pii, normalize_whitespace, truncate_long_text, clean_text

def test_mask_pii_email():
    text = "Contact me at john.doe@example.com"
    result = mask_pii(text)
    assert "[EMAIL]" in result

def test_mask_pii_phone():
    text = "My number is 9876543210"
    result = mask_pii(text)
    assert "[PHONE]" in result

def test_normalize_whitespace():
    text = "Hello   world\nthis is   test"
    result = normalize_whitespace(text)
    assert result == "Hello world this is test"

def test_truncate_long_text():
    long_text = "A" * 4000
    result = truncate_long_text(long_text)
    assert len(result) <= 2000
    assert "[TRUNCATED]" in result

def test_clean_text_combines_all():
    text = "Hello  world\nContact me at abc@gmail.com 9876543210"
    result = clean_text(text)
    assert "[EMAIL]" in result and "[PHONE]" in result
    assert "  " not in result

def test_empty_input_safe():
    assert clean_text("") == ""
    assert mask_pii(None) is None
