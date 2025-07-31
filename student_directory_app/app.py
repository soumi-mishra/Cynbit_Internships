import streamlit as st
import pandas as pd

st.title("🎓 Student Directory App")

# Initialize session state
if 'students' not in st.session_state:
    st.session_state.students = []

# --- Add Student Form ---
st.header("📥 Add a New Student")

with st.form("add_student"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    course = st.text_input("Course")
    score = st.number_input("Score", min_value=0, max_value=100, step=1)

    submitted = st.form_submit_button("Submit")
    if submitted:
        if name and email and course:
            st.session_state.students.append({
                "Name": name,
                "Email": email,
                "Course": course,
                "Score": score
            })
            st.success("✅ Student added!")
        else:
            st.error("❌ Please fill all fields.")

# Convert to DataFrame
df = pd.DataFrame(st.session_state.students)

# --- Display Students ---
st.header("📊 All Students")
if not df.empty:
    st.dataframe(df)

    # --- Remove Student ---
    st.subheader("🗑️ Remove Student")
    remove_index = st.selectbox(
        "Select student to remove:",
        options=df.index,
        format_func=lambda i: f"{df.at[i, 'Name']} ({df.at[i, 'Email']})"
    )
    if st.button("Remove Selected Student"):
        st.session_state.students.pop(remove_index)
        st.success("🗑️ Student removed!")
        st.experimental_rerun()

    # --- Update Student ---
    st.subheader("✏️ Update Student")
    update_index = st.selectbox(
        "Select student to update:",
        options=df.index,
        format_func=lambda i: f"{df.at[i, 'Name']} ({df.at[i, 'Email']})",
        key="update_select"
    )

    student_to_update = st.session_state.students[update_index]
    with st.form("update_form"):
        new_name = st.text_input("New Name", value=student_to_update["Name"])
        new_email = st.text_input("New Email", value=student_to_update["Email"])
        new_course = st.text_input("New Course", value=student_to_update["Course"])
        new_score = st.number_input("New Score", min_value=0, max_value=100, step=1, value=student_to_update["Score"])

        updated = st.form_submit_button("Update Student")
        if updated:
            st.session_state.students[update_index] = {
                "Name": new_name,
                "Email": new_email,
                "Course": new_course,
                "Score": new_score
            }
            st.success("✅ Student updated successfully!")
            st.experimental_rerun()

    # --- Filter Students ---
    st.subheader("🔍 Filter Students")
    filter_type = st.radio("Filter by:", ["None", "Course", "Score ≥"])

    if filter_type == "Course":
        selected_course = st.selectbox("Select Course", df['Course'].unique())
        filtered_df = df[df['Course'] == selected_course]
        st.write(f"Students in **{selected_course}**:")
        st.dataframe(filtered_df)

    elif filter_type == "Score ≥":
        min_score = st.slider("Minimum Score", 0, 100, 50)
        filtered_df = df[df['Score'] >= min_score]
        st.write(f"Students with score ≥ {min_score}:")
        st.dataframe(filtered_df)

else:
    st.info("No students added yet.")