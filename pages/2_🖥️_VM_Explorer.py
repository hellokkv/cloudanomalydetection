import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🖥️ VM Explorer")

data = pd.DataFrame(st.session_state.history)

vm_list = data["vm_id"].unique()
vm = st.selectbox("Select VM", vm_list)

vm_df = data[data["vm_id"] == vm].tail(100)

st.write(f"### 📊 Performance Trends for {vm}")

col1, col2 = st.columns(2)

with col1:
    fig = px.line(vm_df, x="timestamp", y="cpu_usage", title="CPU Usage")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.line(vm_df, x="timestamp", y="memory_usage", title="Memory Usage")
    st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(
    vm_df,
    x="timestamp",
    y="network_traffic",
    title="Network Traffic",
    color_discrete_sequence=["green"]
)
st.plotly_chart(fig2, use_container_width=True)

st.write("### Latest VM Records")
st.dataframe(vm_df.tail(20), use_container_width=True)
