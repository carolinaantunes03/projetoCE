import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import statistics
from itertools import cycle  # To cycle through colors

# Set default template for all Plotly figures
pio.templates.default = "plotly_white"

# --- Configuration ---
# Base results directory
BASE_RESULTS_DIR = "results"

# --- Helper Functions ---

# ... (load_experiment_data, load_experiment_config, plot_history, plot_multiple_histories remain the same) ...


def load_experiment_data(experiment_folder):
    """Loads data from all runs within a selected experiment folder."""
    all_run_data = []
    run_folders = []
    if not os.path.isdir(experiment_folder):
        st.error(f"Experiment folder not found: {experiment_folder}")
        return []
    try:
        run_folders = [f for f in os.listdir(experiment_folder) if f.startswith(
            "run_") and os.path.isdir(os.path.join(experiment_folder, f))]
        # Sort runs numerically
        run_folders.sort(key=lambda x: int(x.split('_')[1]))
    except Exception as e:
        st.error(f"Error listing runs in {experiment_folder}: {e}")
        return []

    for run_folder_name in run_folders:
        run_path = os.path.join(experiment_folder, run_folder_name)
        json_path = os.path.join(run_path, "run.json")
        gif_path = os.path.join(run_path, "best_robot.gif")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    data['run_name'] = run_folder_name
                    data['gif_path'] = gif_path if os.path.exists(
                        gif_path) else None
                    all_run_data.append(data)
            except json.JSONDecodeError:
                st.warning(f"Could not decode JSON for {run_folder_name}")
            except Exception as e:
                st.error(f"Error loading data for {run_folder_name}: {e}")
    return all_run_data


def load_experiment_config(experiment_folder):
    """Loads the main experiment configuration file."""
    config_path = os.path.join(experiment_folder, "experiment.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return config_data
        except json.JSONDecodeError:
            st.warning(
                f"Could not decode experiment.json in {experiment_folder}")
        except Exception as e:
            st.error(f"Error loading experiment.json: {e}")
    else:
        # Be less verbose if it's just not found during initial selection steps
        # st.warning(f"experiment.json not found in {experiment_folder}")
        pass
    return None


def plot_history(all_run_data, history_key, title):
    """Creates a Plotly line chart for a given history key (e.g., 'best_fitness_history')."""
    all_histories = []
    min_len = float('inf')
    for run_data in all_run_data:
        history = run_data.get(history_key)
        if history:
            all_histories.append(history)
            min_len = min(min_len, len(history))

    if not all_histories:
        # st.warning(f"No data found for '{history_key}'") # Less verbose
        return None

    # Truncate histories to the minimum length for aggregation
    truncated_histories = [h[:min_len] for h in all_histories]

    # Transpose so generations are rows
    df = pd.DataFrame(truncated_histories).T
    df.index.name = 'Generation'

    mean_values = df.mean(axis=1)
    std_values = df.std(axis=1)
    # Handle potential NaN std dev if only one run
    std_values = std_values.fillna(0)
    upper_bound = mean_values + std_values
    lower_bound = mean_values - std_values

    fig = go.Figure()

    # Add shaded area for std dev - lighter color for white background
    fill_color = 'rgba(0,100,80,0.2)'  # Example: Teal fill
    fig.add_trace(go.Scatter(
        x=df.index,
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        fillcolor=fill_color,
        fill='tonexty',
        showlegend=False,
        name='Std Dev Upper'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=lower_bound,
        mode='lines',
        line=dict(width=0),
        fillcolor=fill_color,
        fill='tozeroy',
        showlegend=False,
        name='Std Dev Lower'
    ))

    # Add mean line - thicker and distinct color
    line_color = 'rgb(0,100,80)'  # Matching Teal line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=mean_values,
        mode='lines',
        line=dict(color=line_color, width=2.5),  # Increased width
        name='Mean'  # This name might need adjustment if used with plot_multiple_histories
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,  # Center title
            xanchor='center',
            font=dict(size=18)  # Larger title font
        ),
        xaxis_title=dict(
            text="Generation",
            font=dict(size=14)  # Larger axis label font
        ),
        yaxis_title=dict(
            text="Value",
            font=dict(size=14)),
        hovermode="x unified",
        legend=dict(yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    font=dict(size=12)
                    ),
        xaxis=dict(showgrid=True,
                   gridwidth=1,
                   gridcolor='LightGray'
                   ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        margin=dict(l=60, r=30, t=60, b=50))
    return fig


def load_all_experiments_summary(task, env):
    """
    Load summary metrics for all experiments in a task/environment combination
    """
    summary_data = []
    env_path = os.path.join(BASE_RESULTS_DIR, task, env)

    try:
        experiment_folders = [d for d in os.listdir(env_path)
                              if os.path.isdir(os.path.join(env_path, d))]
        # Sort by name, newest first if timestamped
        experiment_folders.sort(reverse=True)
    except Exception as e:
        st.error(f"Error listing experiments in {env_path}: {e}")
        return []

    for exp_name in experiment_folders:
        exp_path = os.path.join(env_path, exp_name)
        exp_data = load_experiment_data(exp_path)

        if exp_data:
            # Calculate summary metrics
            num_runs = len(exp_data)

            # Extract final best fitness and reward values
            final_best_fitness = [run['best_fitness_history'][-1]
                                  for run in exp_data if run.get('best_fitness_history')]
            final_best_reward = [run['best_reward_history'][-1]
                                 for run in exp_data if run.get('best_reward_history')]

            # Calculate statistics if data is available
            fitness_mean = statistics.mean(
                final_best_fitness) if final_best_fitness else None
            fitness_std = statistics.stdev(final_best_fitness) if final_best_fitness and len(
                final_best_fitness) > 1 else 0

            reward_mean = statistics.mean(
                final_best_reward) if final_best_reward else None
            reward_std = statistics.stdev(final_best_reward) if final_best_reward and len(
                final_best_reward) > 1 else 0

            # Add to summary data
            summary_data.append({
                'Experiment': exp_name,
                'Runs': num_runs,
                'Best Fitness (Mean)': fitness_mean,
                'Best Fitness (Std)': fitness_std,
                'Best Reward (Mean)': reward_mean,
                'Best Reward (Std)': reward_std,
            })

    return summary_data


def plot_multiple_histories(all_run_data, history_keys_map, title):
    """
    Creates a Plotly line chart comparing multiple history keys.
    history_keys_map: A dictionary mapping history_key to display name (e.g., {'best_fitness_history': 'Best Fitness'})
    """
    fig = go.Figure()
    # Define a color cycle for lines and fills
    color_palette = px.colors.qualitative.Plotly  # Or choose another palette
    color_cycle = cycle(color_palette)
    fill_opacity = 0.15  # Lower opacity for overlapping areas

    common_min_len = float('inf')
    processed_data = {}

    # First pass: find the minimum length across all relevant histories
    for history_key in history_keys_map.keys():
        all_histories = []
        for run_data in all_run_data:
            history = run_data.get(history_key)
            if history:
                all_histories.append(history)
                common_min_len = min(common_min_len, len(history))
        if not all_histories:
            # st.warning(f"No data found for '{history_key}' in plot_multiple_histories") # Less verbose
            # Decide how to handle this - skip this key or return None?
            # For now, let's skip this key if no data found
            continue
        processed_data[history_key] = {'histories': all_histories}

    if not processed_data or common_min_len == float('inf'):
        # st.warning("No data found for any specified keys in plot_multiple_histories.") # Less verbose
        return None

    # Second pass: process data and add traces
    for history_key, display_name in history_keys_map.items():
        if history_key not in processed_data:
            continue  # Skip if no data was found in the first pass

        all_histories = processed_data[history_key]['histories']

        # Truncate histories to the common minimum length
        truncated_histories = [h[:common_min_len] for h in all_histories]

        df = pd.DataFrame(truncated_histories).T
        df.index.name = 'Generation'

        mean_values = df.mean(axis=1)
        std_values = df.std(axis=1).fillna(0)
        upper_bound = mean_values + std_values
        lower_bound = mean_values - std_values

        # Get next color from the cycle
        line_color_rgb = next(color_cycle)
        # Convert Plotly hex/named color to rgba for fill
        try:
            # Use Plotly's internal conversion if possible (may vary with version)
            from plotly.colors import unlabel_rgb, label_rgb
            rgba_tuple = unlabel_rgb(line_color_rgb)
            fill_color_rgba = f'rgba({rgba_tuple[0]},{rgba_tuple[1]},{rgba_tuple[2]},{fill_opacity})'
        except:  # Fallback for simpler conversion if above fails
            # Basic check for rgb format
            if line_color_rgb.startswith('rgb('):
                fill_color_rgba = line_color_rgb.replace(
                    'rgb(', 'rgba(').replace(')', f',{fill_opacity})')
            else:  # Assume hex or named color - less reliable conversion
                # This is a placeholder - proper hex->rgba conversion is more involved
                fill_color_rgba = 'rgba(128,128,128,0.1)'  # Default grey fill

        # Add shaded area for std dev
        fig.add_trace(go.Scatter(
            x=df.index, y=upper_bound, mode='lines', line=dict(width=0),
            fillcolor=fill_color_rgba, fill='tonexty', showlegend=False, name=f'{display_name} Std Dev Upper',
            hoverinfo='skip'  # Don't show hover for bounds
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=lower_bound, mode='lines', line=dict(width=0),
            fillcolor=fill_color_rgba, fill='tozeroy', showlegend=False, name=f'{display_name} Std Dev Lower',
            hoverinfo='skip'  # Don't show hover for bounds
        ))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=df.index, y=mean_values, mode='lines',
            line=dict(color=line_color_rgb, width=2.5),
            name=display_name  # Use the display name for the legend
        ))

    # Common layout settings
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
        xaxis_title=dict(text="Generation", font=dict(size=14)),
        yaxis_title=dict(text="Fitness", font=dict(
            size=14)),  # Make y-axis specific
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left",
                    x=0.01, font=dict(size=12)),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        margin=dict(l=60, r=30, t=60, b=50)
    )
    return fig


# --- Streamlit App ---
st.set_page_config(layout="wide")
# st.title("EvoGym Experiment Dashboard") # Removed title for space

# --- Sidebar ---
st.sidebar.header("Experiment Selection")

# --- Hierarchical Selection ---
selected_task = None
selected_env = None
selected_experiment_name = None

# 1. Select Task
try:
    tasks = [d for d in os.listdir(BASE_RESULTS_DIR) if os.path.isdir(
        os.path.join(BASE_RESULTS_DIR, d))]
    tasks.sort()
except FileNotFoundError:
    st.sidebar.error(f"Base results directory not found: {BASE_RESULTS_DIR}")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error listing tasks: {e}")
    st.stop()

if not tasks:
    st.sidebar.warning("No task folders found.")
    st.stop()

selected_task = st.sidebar.selectbox("1. Choose Task:", tasks)

# 2. Select Environment
if selected_task:
    task_path = os.path.join(BASE_RESULTS_DIR, selected_task)
    try:
        environments = [d for d in os.listdir(
            task_path) if os.path.isdir(os.path.join(task_path, d))]
        environments.sort()
    except Exception as e:
        st.sidebar.error(f"Error listing environments in {selected_task}: {e}")
        st.stop()

    if not environments:
        st.sidebar.warning(
            f"No environment folders found in task '{selected_task}'.")
        st.stop()

    selected_env = st.sidebar.selectbox("2. Choose Environment:", environments)

# 3. Select Experiment
if selected_task and selected_env:
    env_path = os.path.join(BASE_RESULTS_DIR, selected_task, selected_env)
    try:
        experiment_folders = [d for d in os.listdir(
            env_path) if os.path.isdir(os.path.join(env_path, d))]
        # Sort by name, potentially reverse for newest first if names include timestamps
        experiment_folders.sort(reverse=True)
    except Exception as e:
        st.sidebar.error(f"Error listing experiments in {selected_env}: {e}")
        st.stop()

    if not experiment_folders:
        st.sidebar.warning(
            f"No experiment folders found in environment '{selected_env}'.")
        st.stop()

    selected_experiment_name = st.sidebar.selectbox(
        "3. Choose Experiment:", experiment_folders)


# --- Main Area ---
# --- Main Area ---
if selected_task and selected_env:
    st.header(f"Results Summary: {selected_task} / {selected_env}")

    # Load and display summary of all experiments
    with st.spinner(f"Loading experiment summaries for {selected_task}/{selected_env}..."):
        summary_data = load_all_experiments_summary(
            selected_task, selected_env)

    if summary_data:
        st.subheader("All Experiments Summary")

        # Convert data to pandas DataFrame for better display and sorting
        df = pd.DataFrame(summary_data)

        # Format numeric columns
        for col in ['Best Fitness (Mean)', 'Best Fitness (Std)', 'Best Reward (Mean)', 'Best Reward (Std)']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{x:.4f}" if x is not None else "N/A")

        # Add a formatted column combining mean ± std for compactness
        if 'Best Fitness (Mean)' in df.columns and 'Best Fitness (Std)' in df.columns:
            df['Avg. Best Fitness (± std)'] = df.apply(
                lambda row: f"{row['Best Fitness (Mean)']} ± {row['Best Fitness (Std)']}"
                if row['Best Fitness (Mean)'] != "N/A" else "N/A", axis=1)

        if 'Best Reward (Mean)' in df.columns and 'Best Reward (Std)' in df.columns:
            df['Avg. Best Reward (± std)'] = df.apply(
                lambda row: f"{row['Best Reward (Mean)']} ± {row['Best Reward (Std)']}"
                if row['Best Reward (Mean)'] != "N/A" else "N/A", axis=1)

        # Select columns for display
        display_cols = ['Experiment', 'Runs',
                        'Avg. Best Fitness (± std)', 'Avg. Best Reward (± std)']
        display_df = df[display_cols]

        # Display the table with formatting
        st.dataframe(display_df, use_container_width=True)

        # Add a selection mechanism below the table
        st.write("**Select an experiment from the table to view details:**")
        experiment_options = df['Experiment'].tolist()
        table_selected_experiment = st.selectbox(
            "Choose experiment:",
            options=experiment_options,
            key="table_experiment_selector",
            label_visibility="collapsed"
        )

        # If an experiment is selected from the dropdown, use it
        if table_selected_experiment:
            # Use this selection for the detailed view below
            selected_experiment_name = table_selected_experiment

            # Display experiment details below
            if selected_task and selected_env and selected_experiment_name:
                experiment_path = os.path.join(
                    BASE_RESULTS_DIR, selected_task, selected_env, selected_experiment_name)

                # Load experiment data
                with st.spinner(f"Loading data for {selected_experiment_name}..."):
                    experiment_runs_data = load_experiment_data(
                        experiment_path)
                    experiment_config = load_experiment_config(experiment_path)

    else:
        st.info("No experiment data found.")


# Load experiment-specific data only when an experiment is selected
if selected_task and selected_env and selected_experiment_name:
    experiment_path = os.path.join(
        BASE_RESULTS_DIR, selected_task, selected_env, selected_experiment_name)

    # Load experiment data
    with st.spinner(f"Loading data for {selected_experiment_name}..."):
        experiment_runs_data = load_experiment_data(experiment_path)
        experiment_config = load_experiment_config(experiment_path)

    # Display Experiment Configuration if loaded
    if experiment_config:
        st.subheader("Experiment Configuration")
        with st.expander("View Configuration Details"):
            st.json(experiment_config)  # Display the config as a JSON object
        st.markdown("---")  # Add a separator

    # Now we can safely use experiment_runs_data
    if experiment_runs_data:
        # --- Display Metrics ---
        st.subheader("Summary Metrics (Across Runs)")
        final_best_fitness = [run['best_fitness_history'][-1]
                              for run in experiment_runs_data if run.get('best_fitness_history')]
        final_best_reward = [run['best_reward_history'][-1]
                             for run in experiment_runs_data if run.get('best_reward_history')]

        col1, col2, col3 = st.columns(3)
        num_runs = len(experiment_runs_data)
        with col1:
            st.metric("Number of Runs", num_runs)
        with col2:
            if final_best_fitness:
                mean_final_best = statistics.mean(final_best_fitness)
                std_final_best = statistics.stdev(
                    final_best_fitness) if num_runs > 1 else 0
                st.metric("Avg. Best Overall Fitness",
                          f"{mean_final_best:.4f} ± {std_final_best:.4f}")
            else:
                st.metric("Avg. Best Overall Fitness", "N/A")
        with col3:
            if final_best_reward:
                mean_final_reward = statistics.mean(final_best_reward)
                std_final_reward = statistics.stdev(
                    final_best_reward) if num_runs > 1 else 0
                st.metric("Avg. Best Overall Reward",
                          f"{mean_final_reward:.4f} ± {std_final_reward:.4f}")
            else:
                st.metric("Avg. Best Overall Reward", "N/A", delta_color="off")

        # Continue with the rest of the experiment-specific displays...

    # --- Display Plots ---
    # --- Combined Fitness Plot ---
    st.markdown("---")  # Add a separator
    st.subheader("Combined Fitness History")
    combined_fitness_keys = {
        'best_fitness_history': 'Best Fitness',
        'average_fitness_history': 'Average Fitness'
    }
    fig_combined_fitness = plot_multiple_histories(
        experiment_runs_data, combined_fitness_keys, "Best vs Average Fitness per Generation")
    if fig_combined_fitness:
        st.plotly_chart(fig_combined_fitness, use_container_width=True)
    else:
        st.info("No fitness data available to plot.")

    # --- Individual Reward Plots ---
    st.markdown("---")  # Add a separator
    st.subheader("Reward Metrics")
    fig_best_reward = plot_history(
        experiment_runs_data, 'best_reward_history', "Best Reward per Generation")
    fig_avg_reward = plot_history(
        experiment_runs_data, 'average_reward_history', "Average Reward per Generation")

    col_plots1, col_plots2 = st.columns(2)

    with col_plots1:
        if fig_best_reward:
            st.plotly_chart(fig_best_reward, use_container_width=True)
        else:
            st.caption("No best reward data.")  # Placeholder if no plot
    with col_plots2:
        if fig_avg_reward:
            st.plotly_chart(fig_avg_reward, use_container_width=True)
        else:
            st.caption("No average reward data.")  # Placeholder if no plot

        # --- Display GIFs ---
    st.markdown("---")  # Add a separator
    st.subheader("Best Robots per Run")

    # First find the best run (highest reward)
    best_run = None
    best_reward = float('-inf')

    for run_data in experiment_runs_data:
        if run_data.get('gif_path') and run_data.get('best_reward_history'):
            final_reward = run_data['best_reward_history'][-1]
            if final_reward > best_reward:
                best_reward = final_reward
                best_run = run_data

    # Display the best run GIF prominently
        # Display the best run GIF prominently
    if best_run and best_run.get('gif_path'):
        st.markdown("### Best Overall Robot")
        st.write(f"**{best_run['run_name']}**")
        try:
            # Read GIF file as binary data
            with open(best_run['gif_path'], 'rb') as f:
                gif_data = f.read()

            # Use base64 encoding and HTML to ensure animation works
            import base64
            b64 = base64.b64encode(gif_data).decode()

            # Display using HTML with proper styling
            html = f"""
                <div style="display: flex; justify-content: center;">
                    <img src="data:image/gif;base64,{b64}" width="400" alt="Best Robot Animation">
                </div>
            """
            st.markdown(html, unsafe_allow_html=True)

            # Display both fitness and reward for the best run
            best_fitness = best_run.get('best_fitness_history', [0])[-1]
            best_reward = best_run.get('best_reward_history', [0])[-1]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Fitness", f"{best_fitness:.4f}")
            with col2:
                st.metric("Best Reward", f"{best_reward:.4f}")

        except Exception as e:
            st.warning(f"Could not load best GIF: {e}")

    # Display all GIFs in a grid
    st.markdown("### All Robots")
    num_cols = 5  # Adjust number of columns for GIFs
    cols = st.columns(num_cols)
    col_idx = 0
    gif_found = False

    for i, run_data in enumerate(experiment_runs_data):
        if run_data.get('gif_path'):
            gif_found = True
            with cols[col_idx % num_cols]:
                st.write(f"**{run_data['run_name']}**")
                try:
                    # Read and display GIF directly
                    with open(run_data['gif_path'], 'rb') as f:
                        gif_data = f.read()
                    st.image(gif_data)

                    # Display both fitness and reward for this run
                    if run_data.get('best_fitness_history'):
                        best_fitness = max(
                            run_data['best_fitness_history'], default=0)
                        st.caption(f"Fitness: {best_fitness:.4f}")

                    if run_data.get('best_reward_history'):
                        final_reward = run_data['best_reward_history'][-1]
                        st.caption(f"Reward: {final_reward:.4f}")

                except FileNotFoundError:
                    st.warning(f"GIF not found for {run_data['run_name']}")
                except Exception as e:
                    st.warning(
                        f"Could not load GIF for {run_data['run_name']}: {e}")
            col_idx += 1

    if not gif_found:
        st.info("No GIFs found for this experiment.")


else:
    st.info(
        "Select a task, environment, and experiment from the sidebar to view results.")

# --- How to Run ---
# 1. Save this code as visualization_dashboard.py in your project root (c:\Users\josem\uc\ce\projetoCE\).
# 2. Make sure you have the necessary libraries: pip install streamlit pandas plotly Pillow
# 3. Open your terminal in the project directory.
# 4. Run: streamlit run visualization_dashboard.py
