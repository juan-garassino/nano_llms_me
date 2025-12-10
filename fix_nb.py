import json
import os

notebook_path = 'notebooks/Train_Shakespeare.ipynb'

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Find the cell
    target_source_snippet = "from nano_moe.config import TrainingConfig"
    new_source_code = [
        "import os\n",
        "import sys\n",
        "\n",
        "# Add parent directory to path\n",
        "current_dir = os.getcwd()\n",
        "\n",
        "print(f\"Current Working Directory: {current_dir}\")\n",
        "print(\"Files in current directory:\")\n",
        "print(os.listdir(current_dir))\n",
        "\n",
        "def setup_environment():\n",
        "    # Check if running in Colab\n",
        "    try:\n",
        "        import google.colab\n",
        "        IN_COLAB = True\n",
        "    except ImportError:\n",
        "        IN_COLAB = False\n",
        "\n",
        "    if IN_COLAB:\n",
        "        print(\"\\n[INFO] Detected Google Colab environment.\")\n",
        "        \n",
        "        # Check if we're in /content and the repo is cloned\n",
        "        cwd = os.getcwd()\n",
        "        if cwd == '/content' and os.path.exists('nano_llms_me'):\n",
        "            print(\"[INFO] Found cloned repo 'nano_llms_me'. Changing directory...\")\n",
        "            os.chdir('nano_llms_me')\n",
        "            cwd = os.getcwd()\n",
        "            print(f\"[INFO] New working directory: {cwd}\")\n",
        "        \n",
        "        # Check if nano_moe is present\n",
        "        if not os.path.exists('nano_moe'):\n",
        "            print(\"[WARNING] 'nano_moe' package NOT found in current directory.\")\n",
        "            print(\"You are likely connected to a remote runtime that does not have your local files.\")\n",
        "            print(\"Please upload 'nano_moe.zip' now.\")\n",
        "            \n",
        "            try:\n",
        "                from google.colab import files\n",
        "                uploaded = files.upload()\n",
        "                if 'nano_moe.zip' in uploaded:\n",
        "                    import zipfile\n",
        "                    with zipfile.ZipFile('nano_moe.zip', 'r') as zip_ref:\n",
        "                        zip_ref.extractall('.')\n",
        "                    print(\"[SUCCESS] Extracted nano_moe.zip\")\n",
        "            except Exception as e:\n",
        "                print(f\"[ERROR] Upload widget failed: {e}\")\n",
        "                print(\"Try cloning the repo: !git clone https://github.com/YOUR_USERNAME/nano_llms_me.git\")\n",
        "        else:\n",
        "            print(\"[INFO] 'nano_moe' package found.\")\n",
        "        \n",
        "        # Add current dir to path to find the extracted package\n",
        "        cwd = os.getcwd()\n",
        "        if cwd not in sys.path:\n",
        "            sys.path.append(cwd)\n",
        "\n",
        "    else:\n",
        "        # Local development\n",
        "        project_root = os.path.abspath(os.path.join(current_dir, '..'))\n",
        "        if project_root not in sys.path:\n",
        "            sys.path.append(project_root)\n",
        "\n",
        "setup_environment()\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import numpy as np\n",
        "from rich.console import Console\n",
        "\n",
        "try:\n",
        "    import nano_moe\n",
        "    print(f\"[SUCCESS] Successfully imported nano_moe from {nano_moe.__file__}\")\n",
        "except ImportError:\n",
        "    print(\"[CRITICAL] Could not import nano_moe. Please check the output above.\")\n",
        "\n",
        "from nano_moe.config import TrainingConfig\n",
        "from nano_moe.data.text import get_text_loaders\n",
        "from nano_moe.models.phase import SparseFourierPhaseTransformer\n",
        "from nano_moe.training.trainer import train_epoch, eval_model\n",
        "from nano_moe.training.tracker import ExperimentTracker\n",
        "\n",
        "console = Console()"
    ]

    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Join source to check for the snippet
            source_text = "".join(source)
            if target_source_snippet in source_text:
                cell['source'] = new_source_code
                found = True
                break

    if found:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print("Target cell not found.")

except Exception as e:
    print(f"Error: {e}")
