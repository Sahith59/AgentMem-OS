# Conversation context for calendar operation classification

This four-way control uses current text and a preceding exchange to predict create, update, delete or other on SMCalFlow. The target is a preflight wrapper in the annotated program. Execution, target identity and corrected field values are separate tasks. See PROTOCOL.md for data partitions, exclusions, metrics and controls.

Run `python run.py /path/to/output --fetch` with numpy, scipy and scikit-learn. The program downloads only the pinned official training and validation archives and checks SHA-256. Original data are distributed separately by Microsoft under CC BY-SA 4.0. See the [dataset and paper](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis). Input and derived files go into the directory supplied on the command line.

On 13,336 development turns from 3,424 dialogues, current user text alone scored 82.50 macro-F1. Adding the previous user and assistant exchange scored 87.23. The paired difference was +4.72 points, with a 95% whole-dialogue bootstrap interval of +3.92 to +5.44. The previous-exchange model used 2.68 times as many nonzero input features under the same training passes. Common assistant reply templates account for a substantial share of this gain. This result measures an auxiliary classification baseline.
