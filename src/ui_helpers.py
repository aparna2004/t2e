def metric_card_html(title: str, value) -> str:
    return f"""
    <div style="background: linear-gradient(180deg, rgba(22,22,22,0.94), rgba(33,33,33,0.94)); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 0.9rem 1rem; margin-bottom: 0.75rem;">
        <div style="font-size:0.9rem;color:#b8c0cc;">{title}</div>
        <div style="font-size:1.5rem;font-weight:700;color:#f5f7fb;">{value}</div>
    </div>
    """


def tag_html(text: str) -> str:
    return f"""
    <span style="display:inline-block; padding:0.35rem 0.7rem; margin:0 0.45rem 0.45rem 0; border-radius:999px; background:rgba(59,130,246,0.14); border:1px solid rgba(59,130,246,0.35); color:#dbeafe; font-size:0.92rem;">{text}</span>
    """


def code_block_html(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <pre style="background:#111827; color:#e5e7eb; padding:1rem; border-radius:16px; border:1px solid rgba(255,255,255,0.08); overflow-x:auto; font-size:0.9rem;">{escaped}</pre>
    """
