import os
import shutil
import time
import pandas as pd
import gradio as gr
from pymatgen.core import Structure
import nglview
from utils import get_files_in_working_directory

def get_working_directories():
    base_path = "./data/"
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def on_open_working_directory(working_directory):
    if working_directory is None or working_directory.strip() == "":
        gr.Warning("Please specify a working directory.")
        return gr.update(), None, None, gr.update()
    
    working_directory_path = os.path.join("./data/", working_directory)
    os.makedirs(working_directory_path, exist_ok=True)
    files = get_files_in_working_directory(working_directory_path)
    
    return gr.update(choices=get_working_directories(), value=working_directory), working_directory_path, files, gr.update(interactive=True)

def on_file_list_change(working_directory_path):
    files = get_files_in_working_directory(working_directory_path)
    
    # Update the file dataframe
    file_info = []
    for f in files:
        if f.endswith('.cif'):
            file_type = "Structure file"
        elif f.endswith('INCAR'):
            file_type = "Input File - Calculation parameters"
        elif f.endswith('POSCAR'):
            file_type = "Input File - Coordinates"
        elif f.endswith('KPOINTS'):
            file_type = "Input File - Brillouin-zone sampling"
        elif f.endswith('POTCAR'):
            file_type = "Input File - Pseudopotentials"
        elif f.endswith('vasprun.xml'):
            file_type = "Output File - Summary"
        elif f.endswith('OUTCAR'):
            file_type = "Output File - Summary"
        elif f.endswith('CONTCAR'):
            file_type = "Output File - Coordinates"
        elif f.endswith('OSZICAR'):
            file_type = "Output File - Convergence log"
        elif f.endswith('DOSCAR'):
            file_type = "Output File - Density of States"
        elif f.endswith('CHGCAR'):
            file_type = "Output File - Total charge density"
        elif f.endswith('WAVECAR'):
            file_type = "Output File - Wvefunctions"
        elif f.endswith('EIGENVAL'):
            file_type = "Output File - Eigenvalues"
        else:
            file_type = "Other File"
        
        file_path = os.path.join(working_directory_path, f)
        modified_time = os.path.getmtime(file_path)
        file_info.append([f, file_type, modified_time])
    file_info.sort(key=lambda x: x[2], reverse=True)
    for entry in file_info:
        entry[2] = time.ctime(entry[2])
    file_df = pd.DataFrame(file_info, columns=["File", "Type", "Modified"])

    return file_df

def on_select_file(evt: gr.SelectData):
    selected_file_name = evt.row_value[0]
    if selected_file_name.endswith('.cif') or selected_file_name.endswith('POSCAR') or selected_file_name.endswith('CONTCAR') or selected_file_name.endswith('vasprun.xml') or selected_file_name.endswith('CHGCAR'):
        selected_structure_file = selected_file_name
    else:
        selected_structure_file = None
    if selected_file_name.endswith('.cif') or selected_file_name.endswith('INCAR') or selected_file_name.endswith('POSCAR') or selected_file_name.endswith('KPOINTS') or selected_file_name.endswith('POTCAR') or selected_file_name.endswith('vasprun.xml') or selected_file_name.endswith('OUTCAR') or selected_file_name.endswith('CONTCAR'):
        selected_text_file = selected_file_name
    else:
        selected_text_file = None
    
    return selected_file_name, selected_structure_file, selected_text_file, gr.update(interactive=True)

def on_selected_structure_file_state_change(state):
    return gr.update(interactive=(state is not None))

def on_selected_text_file_state_change(state):
    return gr.update(interactive=(state is not None))

def on_upload_file(working_directory_path, file_path):
    shutil.copy2(file_path, os.path.join(working_directory_path, os.path.basename(file_path)))
    return get_files_in_working_directory(working_directory_path)

def on_delete_file(working_directory_path, selected_file_name):
    if selected_file_name is None:
        return get_files_in_working_directory(working_directory_path)
    
    file_path = os.path.join(working_directory_path, selected_file_name)
    try:
        os.remove(file_path)
        status = "File deleted successfully."
    except Exception as exc:
        status = "Error deleting file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_view_structure_file(working_directory_path, file_name, x, y, z):
    if file_name is None or file_name == "":
        return None
    
    try:
        # Load structure using pymatgen
        file_path = os.path.join(working_directory_path, file_name)
        structure = Structure.from_file(file_path)

        # Make supercell
        cloned_structure = structure.copy()
        supercell = cloned_structure.make_supercell([x, y, z])

        # Create the NGL view widget
        view = nglview.show_pymatgen(supercell)
        view.add_unitcell()
        
        # Write the widget to HTML
        if os.path.exists('./static/input_structure.html'):
            os.remove('./static/input_structure.html')
        nglview.write_html('./static/input_structure.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/input_structure.html?ts={timestamp}" height="500" width="500" title="NGL View"></iframe>'

        return html
                
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None

def on_view_text_file(working_directory_path, text_file_name):
    text_file_path = os.path.join(working_directory_path, text_file_name)
    try:
        with open(text_file_path, 'r') as file:
            content = file.read()
        return gr.update(label=f"Text File Viewer - {text_file_name}", value=content, interactive=True), gr.update(interactive=True)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return gr.update(), gr.update()

def on_save_text_file(working_directory_path, text_file_name, text_content):
    if text_file_name is None:
        gr.Warning("Please select a text file to save.")
        return get_files_in_working_directory(working_directory_path)
    
    text_file_path = os.path.join(working_directory_path, text_file_name)
    try:
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        status = "File saved successfully."
    except Exception as exc:
        status = "Error saving file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def working_directory_blocks():
    with gr.Column(scale=1):
        working_directory_dropdown = gr.Dropdown(label="Working Directory", choices=get_working_directories(), value="wd", allow_custom_value=True)
        working_directory_path_state = gr.State()
        open_working_directory_button = gr.Button(value="Create/Open Working Directory")
        working_directory_file_list_state = gr.State()
        working_directory_file_dataframe = gr.Dataframe(label="Files in Working Directory", headers=["File", "Type", "Modified"], max_height=360, wrap=True, interactive=False)
        selected_file_state = gr.State()
        selected_structure_file_state = gr.State()
        selected_text_file_state = gr.State()
        with gr.Row():
            add_file_upload_button = gr.UploadButton(label="Add File", interactive=False)
            delete_file_button = gr.Button(value="Delete Selected File", interactive=False)
        view_structure_button = gr.Button(value="View Structure File", interactive=False)
        structure_viewer_html = gr.HTML()
        x_input_slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="x")
        y_input_slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="y")
        z_input_slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="z")
        view_text_file_button = gr.Button(value="View Text File", interactive=False)
        text_file_viewer_textarea = gr.TextArea(label="Text File Viewer", lines=20, elem_id="textfile_viewer", interactive=False)
        save_text_file_button = gr.Button(value="Save Text File", interactive=False)
    
    working_directory_dropdown.change(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, working_directory_path_state, working_directory_file_list_state, add_file_upload_button])
    open_working_directory_button.click(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, working_directory_path_state, working_directory_file_list_state, add_file_upload_button])
    working_directory_file_list_state.change(on_file_list_change, working_directory_path_state, working_directory_file_dataframe)
    working_directory_file_dataframe.select(on_select_file, [], [selected_file_state, selected_structure_file_state, selected_text_file_state, delete_file_button])
    selected_structure_file_state.change(on_selected_structure_file_state_change, selected_structure_file_state, view_structure_button)
    selected_text_file_state.change(on_selected_text_file_state_change, selected_text_file_state, view_text_file_button)
    add_file_upload_button.upload(on_upload_file, [working_directory_path_state, add_file_upload_button], working_directory_file_list_state)
    delete_file_button.click(on_delete_file, [working_directory_path_state, selected_file_state], working_directory_file_list_state)
    view_structure_button.click(on_view_structure_file, [working_directory_path_state, selected_structure_file_state, x_input_slider, y_input_slider, z_input_slider], structure_viewer_html)
    x_input_slider.change(on_view_structure_file, [working_directory_path_state, selected_structure_file_state, x_input_slider, y_input_slider, z_input_slider], structure_viewer_html)
    y_input_slider.change(on_view_structure_file, [working_directory_path_state, selected_structure_file_state, x_input_slider, y_input_slider, z_input_slider], structure_viewer_html)
    z_input_slider.change(on_view_structure_file, [working_directory_path_state, selected_structure_file_state, x_input_slider, y_input_slider, z_input_slider], structure_viewer_html)
    view_text_file_button.click(on_view_text_file, [working_directory_path_state, selected_text_file_state], [text_file_viewer_textarea, save_text_file_button])
    save_text_file_button.click(on_save_text_file, [working_directory_path_state, selected_text_file_state, text_file_viewer_textarea], working_directory_file_list_state)

    return working_directory_path_state, working_directory_file_list_state