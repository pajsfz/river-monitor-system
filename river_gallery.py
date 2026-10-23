import streamlit as st


def render_image_gallery(results: list[dict], key_prefix: str) -> dict:
    if not results:
        return st.session_state.result
    index = min(max(st.session_state.image_index, 0), len(results) - 1)
    selected = results[index]
    previous, image_col, next_col = st.columns([1, 10, 1], vertical_alignment="center", gap="small")
    with previous:
        if st.button("", icon=":material/chevron_left:", help="上一张已标框图片", disabled=index == 0, key=f"{key_prefix}_previous"):
            st.session_state.image_index = index - 1
            st.session_state.result = results[index - 1]
            st.session_state.deepseek_text = ""
            st.rerun()
    with image_col:
        st.image(
            selected["annotated_image"],
            caption=f"{index + 1} / {len(results)} · {selected['source_name']}",
            alt="YOLO 已标框图片，可用左右按钮切换",
            width="stretch",
        )
    with next_col:
        if st.button("", icon=":material/chevron_right:", help="下一张已标框图片", disabled=index >= len(results) - 1, key=f"{key_prefix}_next"):
            st.session_state.image_index = index + 1
            st.session_state.result = results[index + 1]
            st.session_state.deepseek_text = ""
            st.rerun()
    return selected
