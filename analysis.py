import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from matplotlib.patches import Rectangle
import re
import os

# Set the font size and family globally
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

colors = ['blue', 'red', 'green', 'yellow', 'purple']

def average_answers_per_question(data):
    """
    Compute the average number of multiple-choice answers per question for each filename.
    """
    results = {}
    
    for filename, content in data.items():
        questions = content.get('questions', [])
        
        total_answers = 0
        for question in questions:
            # Use regex to count the number of multiple-choice answers (e.g., A., B., ...)
            total_answers += len(re.findall(r'\b[A-Z]\.', question))
        
        # Compute the average
        avg_answers = total_answers / len(questions) if questions else 0
        results[filename] = avg_answers
    
    return results

def plot_marks(marks, tests):
    for filename in marks:
        print(f"Analyzing marks results from {filename}")
        
        llm_names = list(marks[filename].keys())
        llm_names.reverse()  # Reverse the order of LLM names
        
        categories = tests[filename]['categories']
        category_names = list(categories.keys())
        num_categories = len(category_names)
        num_llms = len(llm_names)
        bar_width = 0.3

        # Create a concatenated array for all LLMs
        all_marks = []
        for llm_name in llm_names:
            all_marks.append(np.array(marks[filename][llm_name]['marks']).T)
        concatenated_marks = np.concatenate(all_marks, axis=0)
        
        # Create the figure
        fig, ax = plt.subplots(dpi=500)
        handles = []  # To store handles for the legend
        
        for idx, llm_name in enumerate(llm_names):
            print(f"\tAnalyzing results for {llm_name}")
            llm_marks = np.array(marks[filename][llm_name]['marks']).T
            
            # Starting row for the current LLM
            start_row = idx * llm_marks.shape[0]
            
            # Adjust the color index for the reversed LLM order
            color_index = (num_llms - 1) - idx
            
            # Plot individual LLM marks with unique colors using patches
            for i in range(llm_marks.shape[0]):
                for j in range(llm_marks.shape[1]):
                    color = colors[color_index] if llm_marks[i, j] else 'white'
                    rect = Rectangle((j, start_row + i), 1, 1, facecolor=color)
                    ax.add_patch(rect)
                    
            handles.append(Rectangle((0, 0), 1, 1, color=colors[color_index]))

        # After finishing the loop, adjust and save the combined plot
        ax.legend(handles[::-1], llm_names[::-1], loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlim(0, concatenated_marks.shape[1])
        ax.set_xlabel('Question')
        ax.set_ylim(0, concatenated_marks.shape[0])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_ylabel('Trial')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)  # Adjust the right padding to make space for the legend
        plt.savefig(f'figures/{filename[:-4]}_combined_marks.png', bbox_inches='tight')  # Ensure the legend is saved in the image
        plt.close()

def plot_occurrences(marks, tests):
    answers_per_question = average_answers_per_question(tests)
    
    for filename in marks:
        print(f"Analyzing occurrence results from {filename}")

        llm_names = list(marks[filename].keys())
        for llm_name in llm_names:
            print(f"\tAnalyzing results for {llm_name}")
            current_marks = np.array(marks[filename][llm_name]['marks']).T

            # Plot correct answer occurrences
            random_success = []
            for i in np.arange(current_marks.shape[0] + 1):
                random_success.append(poisson_prob(1/(answers_per_question[filename]) * current_marks.shape[0], i) * 100)
            
            plt.figure(dpi=300)
            hist, bin_edges = np.histogram((np.sum(current_marks, axis=0)), bins=np.arange(len(current_marks) + 2) - 0.5)
            scaled_hist = hist / hist.sum() * 100.0
            plt.bar(bin_edges[:-1] + 0.5, scaled_hist, width=np.diff(bin_edges), edgecolor='black', label=llm_name)
            plt.plot(np.arange(current_marks.shape[0] + 1), random_success, marker='o', linestyle='--', color='red', label='Random success')
            plt.xlim(-0.5, len(current_marks) + 0.5)
            plt.xlabel('Correct answer occurrences per-question')
            plt.ylabel('Percentage of questions (%)')
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1])
            plt.savefig(f'figures/{filename[:-4]}_{llm_name}_consistency.png')
            plt.close()

def plot_comparisons(marks, tests):
    for filename in marks:
        print(f"Analyzing comparison results from {filename}")
        
        llm_names = list(marks[filename].keys())
        categories = tests[filename]['categories']

        category_names = list(categories.keys())
        num_categories = len(category_names)
        num_llms = len(llm_names)
        bar_width = (1.0 - 0.1)/float(num_llms)
        
        std_devs_all = []
        avg_corrs_all = []
        avg_scores_all = []

        for category, question_ranges in categories.items():
            print(f"\tAnalyzing results for {category}")
            std_devs = []
            avg_corrs = []
            avg_scores = []

            for llm_name in llm_names:
                print(f"\t\tAnalyzing results for {llm_name}")
                current_marks_for_category = [marks[filename][llm_name]['marks'][q-1] for r in question_ranges for q in range(r[0], r[1]+1)]
                current_marks_for_category = np.array(current_marks_for_category).T

                total_questions = np.prod(current_marks_for_category.shape)
                correct_answers = np.sum(current_marks_for_category)
                average_score = (correct_answers / total_questions) * 100
                
                avg_scores.append(average_score)
                std_devs.append(np.std(current_marks_for_category))
                avg_corrs.append(np.mean(np.corrcoef(current_marks_for_category, rowvar=True)[np.triu(np.ones_like(np.corrcoef(current_marks_for_category, rowvar=True), dtype=bool), k=1)]))
                
            std_devs_all.append(std_devs)
            avg_corrs_all.append(avg_corrs)
            avg_scores_all.append(avg_scores)
            
        # Grouped Bar Graph plotting for Standard Deviation, Average Correlation, and Average Scores
        x = np.arange(len(category_names))
        xlim = (x[0]-bar_width, x[-1]+bar_width*(num_llms+1)) # this is meant to get rid of white-space in bar plots before and after first and last bar respectively (not right)
        
        # Standard Deviation Plot
        fig, ax = plt.subplots(dpi=500)
        for i, (llm_name, color) in enumerate(zip(llm_names, colors)):
            ax.bar(x + i*bar_width, [std_dev[i] for std_dev in std_devs_all], width=bar_width, label=llm_name, color=color, align='center')
            for j, v in enumerate([std_dev[i] for std_dev in std_devs_all]):
                ax.text(j + i*bar_width, v + 0.01, "{:.2f}".format(v), color='black', ha='center', va='bottom', fontsize=5)
        # ax.set_xlim(xlim)
        ax.set_xticks(x + bar_width*(num_llms-1)/2)
        ax.set_xticklabels(category_names, rotation=45, ha='right')
        ax.set_title('Standard deviation - {}'.format(filename.replace('_',' ')[:-4]))
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig(f'figures/{filename[:-4]}_std_dev.png')
        plt.close()

        # Average Correlation Plot
        fig, ax = plt.subplots(dpi=500)
        for i, (llm_name, color) in enumerate(zip(llm_names, colors)):
            ax.bar(x + i*bar_width, [avg_corr[i] for avg_corr in avg_corrs_all], width=bar_width, label=llm_name, color=color, align='center')
            for j, v in enumerate([avg_corr[i] for avg_corr in avg_corrs_all]):
                ax.text(j + i*bar_width, v + 0.01, "{:.2f}".format(v), color='black', ha='center', va='bottom', fontsize=5)
        # ax.set_xlim(xlim)
        ax.set_xticks(x + bar_width*(num_llms-1)/2)
        ax.set_xticklabels(category_names, rotation=45, ha='right')
        ax.set_title('Average correlation - {}'.format(filename.replace('_',' ')[:-4]))
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig(f'figures/{filename[:-4]}_avg_corr.png')
        plt.close()

        # Average Scores Plot
        fig, ax = plt.subplots(dpi=500)
        for i, (llm_name, color) in enumerate(zip(llm_names, colors)):
            ax.bar(x + i*bar_width, [avg_score[i] for avg_score in avg_scores_all], width=bar_width, label=llm_name, color=color, align='center')
            for j, v in enumerate([avg_score[i] for avg_score in avg_scores_all]):
                ax.text(j + i*bar_width, v + 0.01, "{:.0f}".format(v), color='black', ha='center', va='bottom', fontsize=6)
        # ax.set_xlim(xlim)
        ax.set_xticks(x + bar_width*(num_llms-1)/2)
        ax.set_xticklabels(category_names, rotation=45, ha='right')
        ax.set_title('Average scores - {}'.format(filename.replace('_',' ')[:-4]))
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig(f'figures/{filename[:-4]}_avg_scores.png')
        plt.close()
            
def poisson_prob(mean_occurrences, k_value):
    probability = (math.exp(-mean_occurrences) * (mean_occurrences ** k_value)) / math.factorial(k_value)
    return probability

# Load the marks and tests
with open('marks.json', 'r') as file:
    marks_from_disk = json.load(file)
with open('tests.json', 'r') as file:
    tests_from_disk = json.load(file)

if not os.path.exists('./figures'):
    os.makedirs('./figures')
    
# generate the plots
plot_marks(marks_from_disk, tests_from_disk)
plot_occurrences(marks_from_disk, tests_from_disk)
plot_comparisons(marks_from_disk, tests_from_disk)
