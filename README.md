Cereal Bar V2
=============

- [Cereal Bar V2](#cereal-bar-v2)
  - [Intro](#intro)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Installing the Unity client.](#installing-the-unity-client)
    - [Creating a config.](#creating-a-config)
    - [Running the server.](#running-the-server)
    - [Deploying the server to a new machine.](#deploying-the-server-to-a-new-machine)
  - [Development](#development)
    - [Cloning the repository.](#cloning-the-repository)
    - [Download Submodules](#download-submodules)
    - [Installing Dev Package.](#installing-dev-package)
    - [Pre-commit hooks.](#pre-commit-hooks)
    - [Server](#server)
    - [Client](#client)
    - [Client API.](#client-api)
    - [Scenario Rooms](#scenario-rooms)
      - [Creating a scenario.](#creating-a-scenario)
      - [Launching a scenario.](#launching-a-scenario)
      - [Scenario (Map) Editor](#scenario-map-editor)
        - [Requirements](#requirements)
        - [Running the map editor](#running-the-map-editor)
  - [Documentation](#documentation)
  - [Server Endpoints](#server-endpoints)
      - [Password-protected endpoints.](#password-protected-endpoints)
  - [Demonstration Model](#demonstration-model)
  - [Dataset](#dataset)
  - [Resources](#resources)

Intro
-----

Cereal Bar is a two-player web game designed for studying language
understanding agents in collaborative interactions. **This repository contains**
**code for the game, a webapp hosting the game, and various related tools.**

This is a remake of the original Cereal Bar, which can be found [here][0]

包含：游戏的代码，托管游戏的web应用程序，以及各种相关工具.

Installation
------------

The easiest way to install CB2 is via pip. You can install the game with:

```
python -m pip install cb2game
```

Getting Started
---------------

### Installing the Unity client.

You'll need to get a version of the front-end unity client, as this doesn't come with the pip package. You can fetch the latest client released on Github via:

```
python -m cb2game.server.fetch_client
```

### Creating a config.

The server requires a config file to run. We provide a config generator script that walks you through the step of setting up the configuration for your server.
You can run it with:生成config

```
python -m cb2game.server.generate_config
```

This will create a config file in the current directory. 

If you just want a default config, you can just run:使用默认的config

```
python -m cb2game.server.generate_config --all_defaults
```

Which will create `default.yaml` in the current directory.

> 运行结果：
>
> ```
> (cb2) C:\Users\keyang\Desktop\yan0\Agent\cb2>python -m cb2game.server.generate_config
> 
> Http Settings (1/8)
> =============
> Database location override: (default: '')
> 
> =============
> Generating maps can take ~100ms of CPU time. As an optimization, the server will pre-generate a pool of maps while no games are in progress.
> On a local dev machine, you'll want this to be small since you won't be running many games at once. On a production server, you'll want to find the right balance between memory consumption and map generation time.
> Set to zero to disable pregeneration.
> Pregenerated map pool size (default: 50):
> Fog is a visual effect that makes the map harder to see as you get further away from objects. Only the follower sees fogfog_start is how many cells away things begin to get foggy. fog_end is how many cells away things are completely obscured.
> Enter fog_end (default: 20):
> 
> Mapgen Settings (4/8)
> ===============
> Configuring mapgen requires decided on about ~10 parameters, and is pretty open-ended. This can take a while. You can always do it yourself later by editing the config file.
> If you just want the defaults, you can skip all this.
> Map size (width, height):  (25, 25)
> Number of mountains (min, max):  (3, 3)
> Number of cities (min, max):  (3, 4)
> Number of lakes (min, max):  (3, 4)
> CB2 uses path routing to draw paths on the ground between features on the map. The path connection distance is the maximum distance that two objects can be and still have paths routed between them.
> Path connection distance:  (default: 4):
> 
> Client Settings (5/8)
> ===============
> Any exceptions that occur in the unity client are uploaded to the server. This helps us debug issues. You can place a maximum limit on the number of exceptions that are stored in the database (highly recommended).
> "在unity客户端中发生的任何异常都会上传到服务器。这有助于我们调试问题。您可以对数据库中存储的异常数量设置最大限制(强烈建议)"
> Max client exceptions to store (default: 100):
> Best to leave FPS option default. Some low-end laptops may perform better if a low FPS limit is set instead of self-managing (say, 30). -1 means let the browser optimize for device settings.
> "最好让FPS选项默认。如果设定较低的FPS上限，而不是自我管理(例如30)，一些低端笔记本电脑可能会表现得更好。-1表示让浏览器针对设备设置进行优化。"
> Browser FPS limit (default: -1): 30
> 
> Auth Settings (6/8)
> =============
> Do you want to support Google Auth? You'll need to set up a Google Cloud project and OAuth credentials...
> Enable Google Auth? (y/N) y
> Go here and create an OAuth client ID for a Web Application (very fast): https://console.cloud.google.com/apis/credentials/oauthclient
> Then paste the client ID here...
> Google OAuth Client ID: minecb2game
> Some server URLs are password-protected. See here for more info: https://github.com/lil-lab/cb2/wiki/Cb2-Url-Endpoints
> NOTE: Save your password. Passwords are stored in the config as an SHA512 hash. We don't store the password itself, and it's impossible to recover the password from the hash.
> If you forget your password, you'll need to delete the hash in the config file and restart the server.
> Server password (leave blank for no password):
> 
> Lobby Settings (7/8)
> ==============
> Populating lobbies with default lobby set. You can change this later by editing the config file.
> 
> Final Settings (8/8)
> ==============
> Let's name your config. Full name will be <nameprefix>-<timestamp>-autoconf
> Config name prefix: (default: 'noname')
> Write a comment for this config (explain in a few words what this config is for and where it will be deployed): for testing on February 5th.
> Writing config to noname-1707063637-autoconf.yaml
> Config saved to noname-1707063637-autoconf.yaml.
> ```
>
> 

### Running the server.

Once you have a config, you can run the server with:

```
python -m cb2game.server.main --config_filepath noname-1707063637-autoconf.yaml
follower_bots/pretraining_data/cb2-data-base/config/human_model.yaml # human-model查看回放
follower_bots/pretraining_data/cb2-data-base/config/train.yaml # train 数据回放
```

You can now access the game instance at `http://localhost:8080/`

> ```
> (cb2) C:\Users\keyang\Desktop\yan0\Agent\cb2>python -m cb2game.server.main --config_filepath C:\Users\keyang\Desktop\yan0\Agent\cb2\noname-1707063637-autoconf.yaml
> [2024-02-05 00:23:39,625] cb2game.server.config.config INFO [config:ValidateConfig:48] //////////////// Created data directory C:\Users\keyang\AppData\Local\cb2-game-dev\cb2-game-dev ////////////////
> [2024-02-05 00:23:39,625] cb2game.server.config.config WARNING [config:ValidateConfig:53] Record directory C:\Users\keyang\AppData\Local\cb2-game-dev\cb2-game-dev\game_data.db does not exist. This can happen if it's your first time running a new config. The program will just create a database for you.
> [2024-02-05 00:23:39,626] cb2game.server.config.config WARNING [config:ValidateConfig:58] Record directory C:\Users\keyang\AppData\Local\cb2-game-dev\cb2-game-dev\game_records does not exist. This is okay, it's just a sign that the logged network packets may be missing. Or it's your first time running a config.
> [2024-02-05 00:23:39,626] root INFO [main:main:1471] Config file parsed.
> [2024-02-05 00:23:39,626] root INFO [main:main:1472] data prefix:
> [2024-02-05 00:23:39,626] root INFO [main:main:1473] Log directory: C:\Users\keyang\AppData\Local\cb2-game-dev\cb2-game-dev\game_records
> [2024-02-05 00:23:39,627] root INFO [main:main:1474] Assets directory: C:\Users\keyang\AppData\Local\cb2-game-dev\cb2-game-dev\assets
> [2024-02-05 00:23:39,627] root INFO [main:main:1475] Database path: C:\Users\keyang\AppData\Local\cb2-game-dev\cb2-game-dev\game_data.db
> [2024-02-05 00:23:39,631] cb2game.server.schemas.base INFO [base:SetDatabase:30] Pragmas: [('journal_mode', 'wal'), ('cache_size', -65536), ('foreign_keys', '1'), ('synchronous', 'off')]
> ======= Serving on http://0.0.0.0:8080 ======
> Map pool size: 10
> Map pool size: 20
> Map pool size: 30
> Map pool size: 40
> Map pool size: 50
> ```
>
> 
>
> | <img src="C:\Users\keyang\AppData\Roaming\Typora\typora-user-images\image-20240205002601926.png" alt="image-20240205002601926" style="zoom: 50%;" /> | <img src="C:\Users\keyang\AppData\Roaming\Typora\typora-user-images\image-20240205002652276.png" alt="image-20240205002652276" style="zoom: 50%;" /> |
> | ------------------------------------------------------------ | ------------------------------------------------------------ |
>
> 

### Deploying the server to a new machine.

部署 CB2 游戏服务器到新机器上时的步骤，尤其是在 Ubuntu 22.04 LTS 上使用 systemd 服务进行部署。

If you're setting up a web server, you'll want to run CB2 as a daemon. This
provides a few benefits:

- The server will automatically restart if it crashes.
- Logs will be automatically rotated.
- You can start/stop the server with `systemctl`.
- The server will run in the background, and you can log out of the machine without stopping the server.

We provide a script for deploying CB2 as a systemd service on Ubuntu 22.04 LTS:

```
# Install cb2game service.
python3 -m cb2game.deploy install <path-to-config.yaml>

# Fetch cb2game front-end client and install it. Uses latest release on Github
# if no locally built client is specified. Unless you're interested in
# customizing CB2's Unity client, you should leave this blank.
python3 -m cb2game.deploy fetch-client <optional-path-to-local-client>

# Start the service (check localhost:8080 or python -m cb2game.deploy logs to
# verify)
python3 -m cb2game.deploy start

# Check localhost:8080/ in your browser. The server should now be started!
```

Here's some other useful commands:
```
# Update the current cb2game version. Latest if no version specified.
python3 -m cb2game.deploy update-to-version <optional-version>

# See where all files are installed, current cb2game version installed on system.
python3 -m cb2game.deploy info

# Update the config used by the service.
python3 -m cg2game.deploy update_config <path-to-config.yaml>

# Possibly diagnose system issues causing install to fail.
python3 -m cb2game.deploy diagnose

# Restart
python3 -m cb2game.deploy restart

# Stop the service
python3 -m cb2game.deploy stop

# Access service logs.
python3 -m cb2game.deploy logs

# Uninstall.
python3 -m cb2game.deploy uninstall
```


Development
-----------

Here's the instructions if you'd like to setup CB2 for development. This
installs the `cb2game` package in editable mode, so you can make changes to the
code and have them reflected in the server without having to reinstall the
package.

### Cloning the repository.

This repository uses [git-lfs][1]. Event though newer versions of git (>=
2.3.0) can handle this automatically, the .gitattributes file falls back to
git-lfs for all binary files by default. git lfs is required, so make sure to
install and use `git lfs clone` to clone this repository.

### Download Submodules

This repository contains submodules. As such, you need to run this step to
fetch submodules after cloning:

```
cd repo
git submodule init
git submodule update
```

### Installing Dev Package.

CB2 requires `Python 3.9` or higher.

We recommend you setup a virtual environment for the development of CB2. You can do this with:

* Create the venv with: `python3 -m venv <env_name>` (run once).
* Enter the venv with: `source <env_name>/bin/activate`
* Now that you're in a virtual python environment, you can proceed below to install the server in dev mode.

Install the server in editable mode with:

```
# Run from the root of the repo.
python3 -m pip install -e .
```

### Pre-commit hooks.

**预提交钩子通常用于在将代码提交到github代码仓库之前执行一些操作，例如运行代码格式化工具或执行单元测试。这有助于确保代码符合项目的规范，防止在提交之前引入格式错误或破坏测试的代码**。

****Precommit hooks are only required if you plan to contribute code to the**
repository.**  But they're highly recommended, as they'll run formatting tools on
your code before you commit it, and block the commit if there are any failing
unit tests.  If a unit test fails, it means you have broken something, and you
shouldn't be committing it.

Some precommit hooks may require `python3.10` in order to run. You can download
python3.10 from python.org. You don't need it to be the python version used in
your venv or conda environment, or even the system default. It simply needs to
be installed somewhere on your system, and downloading the binary from
python.org shouldn't interfere with any existing python environments. It will just
make `python3.10` available as a binary on the path.

Pre-commits take a long time (1-2m) to run the first time you commit, but they
should be fast (3-4 seconds) after that.

Install pre-commit hooks with

```pre-commit install```

If you don't have pre-commit already, you can get it by refreshing dependencies.

On every commit, your commit will be blocked if any of the hooks defined in `.pre-commit-config.yaml` fail.

### Server

Launch the server on your desktop with:

```
python3 -m cb2game.server.main --config_filepath <path-to-config>
```

To launch the server on a deployment machine, you'll want to use the SystemD
daemon. This can be installed with the `deploy/deploy.sh` script. It makes use
of the special config file `server/config/server-config.yaml`.

When you're done, you can quit the python venv with `deactivate` on the command line.

### Client

CB2 is designed such that most game logic can be modified without having to
recompile the Unity client. However, if you do need to recompile the client,
you'll need to install Unity.

The client is a Unity project developed using Unity `Version 2020.3.xx`. This is contained in the `unity_client/` directory. Once unity is installed, the application should open successfully.

For development purposes, the server may be run locally and the client run directly in the Unity editor. This connects to the server using the default lobby. For deployment, the game is compiled to HTML + WebGL.

The WebGL client can either be compiled from within Unity or from the command line with [build_client.sh](https://github.com/lil-lab/cb2/blob/main/build_client.sh). This launches a headless version of Unity which builds a WebGL client and moves it to the appropriate directory (`server/www/WebGL`) in the server.

> CB2被设计成这样，大多数游戏逻辑都可以修改，而不必修改
> 重新编译Unity客户端。但是，如果你确实需要重新编译客户端，
> 你需要安装Unity。
>
> 客户端是一个使用Unity `Version 2020.3.xx`开发的Unity项目。它包含在`unity_client/`目录中。安装unity后，应用程序应该能成功打开。
>
> 出于开发目的，服务器可以在本地运行，客户端可以直接在Unity编辑器中运行。这将使用默认大厅连接到服务器。在部署时，游戏被编译为HTML + WebGL。
>
> WebGL客户端可以在Unity中编译，也可以在命令行中使用[build_client.sh](https://github.com/lil-lab/cb2/blob/main/build_client.sh)。这将启动一个无头版本的Unity，它将构建一个WebGL客户端并将其移动到服务器中的适当目录(`server/www/WebGL`)。

```
# Before running this script, open it and change the UNITY variable to the path to your Unity executable. 在运行这个脚本之前，打开它并将UNITY变量更改为你的UNITY可执行文件的路径。
./build_client.sh # Unity must be closed before running this.
```

This launches a headless version of Unity which builds a WebGL client and moves it to the appropriate directory (`server/www/WebGL`) in the server. Any pre-existing contents of `server/www/WebGL` are moved to `server/www/OLD_WebGL`.

Upon completion of this command, one may launch the server and access the client via ```localhost:8080/play```.

If you built the client from unity and want to install it, you can run:

```
python3 -m cb2game.server.fetch_client ----local_client_path <path_to_WebGL_dir>
```

### Client API.

This repository contains a client API for writing agents which can interact with CB2. The client API is contained in directory `pyclient/`, which contains a README with further information.

### Scenario Rooms
CB2 contains a scenario room to allow for research that wants to investigate
custom scenarios in a controlled manner. **Scenario rooms are single player**
**(follower role only, currently),** and allow for a script to attach via the Python
API and monitor the game state. The script can at any time load a new map, or
send instructions/feedback just as the leader would. We provide an in-game UI to
turn an existing game into a scenario for later inspection.

> CB2包含一个场景室，允许进行需要调查的研究以受控的方式自定义场景。场景室是单人游戏
> (当前仅限follower角色)，并允许脚本通过Python附加API并监控游戏状态。脚本可以在任何时候加载一个新地图，或者像领导一样发出指示/反馈。**我们提供了一个游戏内的UI将一个现有的游戏变成一个场景以供以后检查**。

#### Creating a scenario.
**You can create a scenario from inside of a game by hitting escape and then "Save**
**Scenario State".** You must be in the `open` lobby to do this.

Access the open lobby via endpoint `/play?lobby_name=open`.

The scenario file itself is a JSON file that you can download. The JSON follows
the schema of the `Scenario` dataclass defined in `src/cb2game/server/messages/scenario.py`.

**Scenarios are currently follower-only.** 　If it wasn't the followers turn when you　created the scenario, then the follower will be unable to move.　Make sure to　edit the scenario file, specifically the `turn` field of the `turn_state` entry,　to equal to the value `1` (follower). You may also want to give the follower a　large number of moves, so that they can move freely about the scenario.

#### Launching a scenario.
You can launch a scenario by entering a room in the scenario lobby. Scenario
rooms are 1 player, and you play as the follower.

Access the scenario lobby via endpoint `/play?lobby_name=scenario-lobby`

Then hit "Join Game". You'll immediately join an empty scenario. Load a scenario
file by hitting esc and clicking on `Upload Scenario State`. If this item
doesn't appear in the escape menu, reload the page and retry (this sometimes happens).

The scenario should then load. If the file is invalid, then the server will end
the game immediately.

#### Scenario (Map) Editor 通过UI界面编辑自定义的MAP

CB2 contains a map editor, which you can use to craft custom maps. These maps
can be explored in a custom scenario.

##### Requirements
The map editor requires that tkinter is installed on your system. If you didn't
do this prior to setting up your virtual environment, you'll need to install
tkinter, and then re-create your venv (should only take a few minutes --
deleting venv/ is a relatively safe operation)

OSX
```
brew install python-tk
```

Ubuntu
```
sudo apt-get install python-tk python3-tk tk-dev
```

##### Running the map editor

Launch the map editor with the command:

```
# Must be in python virtual env first!
python -m cb2game.server.map_tools.map_editor
```

No further command line parameters are needed. The editor will pop-up a GUI
asking you for a scenario file. We recommend starting with the template map, a
10x10 environment included in this repository at
`src/cb2game/server/map_tools/maps/template.json`.

Upon closing the editor, it pops up another GUI to save the
modified scenario -- Make sure to do this, or your changes will be lost. Hitting
Q or Escape will close the editor, so be careful!

There's currently no undo. If you made a change you want to undo, close the
editor without saving, and then reload the scenario file.

The green button in the UI is to save & quit.
The red button in the UI clears the screen and replaces all tiles with green tiles.

You can resize a scenario map by editing the "rows" and "cols" fields respectively
of the scenario file with a text editor.

> 关闭编辑器后，将弹出另一个GUI来保存
> 修改场景——确保这样做，否则您的更改将丢失。打
> Q或Escape将关闭编辑器，所以要小心!
>
> 目前还没有撤销。如果你想撤销所做的更改，请关闭
> 编辑器，然后重新加载场景文件。
>
> UI中的绿色按钮是保存&退出按钮。
> UI中的红色按钮会清除屏幕并将所有瓷砖替换为绿色
> 瓷砖。
>
> 您可以通过分别编辑“rows”和“cols”字段来调整场景映射的大小
> 场景文件的文本编辑器。

Documentation
-------------
For more information on CB2, see the [CB2 Wiki](https://github.com/lil-lab/cb2/wiki).

Server Endpoints
----------------

The CB2 server creates a number of HTTP endpoints for inspecting and accessing user data from a live server instance. This makes it easier to administer experiments with CB2 – server admins can inspect games in-progress live.

> CB2服务器创建许多HTTP端点，用于从活动服务器实例中检查和访问用户数据。这使得使用CB2管理实验变得更容易——服务器管理员可以实时检查正在进行的游戏。

| Endpoint URL         | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `/`                  | CB2 Homepage. Contains links to docs, code, etc.      |
| `/play`              | Serves Unity WebGL client                             |
| `/player_endpoint`   | Websocket endpoint for communication with clients.    |
| `/view/games`        | View all games played on the server.                  |

For a full list of endpoints and more info, see the [CB2 URLs doc](https://github.com/lil-lab/cb2/wiki/Cb2-Url-Endpoints) in the wiki.

#### Password-protected endpoints.
The server contains some optionally password-protected endpoints. These are endpoints which allow access to game data or live user information. **You can set the password in the config via the server_password_sha512 field.** 

![image-20240222210604331](C:\Users\keyang\AppData\Roaming\Typora\typora-user-images\image-20240222210604331.png)

Do not put the plain text password in your configuration file. Instead, you use a sha512 hash of the password. You can generate a password hash with the following command:

```
python -c 'import hashlib; print(hashlib.sha512(b"your_password").hexdigest())'
```

To access password-protected endpoints, you must pass the password as a query
parameter. For example, if your password is `password`, you would access the
`/view/games` endpoint with the following URL:

```
http://localhost:8080/view/games?password=password
```


Demonstration Model
-------------------

We trained and deployed a baseline demonstration model that is publicly
available online.  You can play against the model on our website, at
[cb2.ai/][2]. For more information on the model, including a link to download
the weights, see the readme at `follower_bots/README.md`.

> 我们训练和部署了一个公开的基线演示模型在网上。你可以在我们的网站上与模型对战
> [cb2.ai /][2]。有关该模型的更多信息，包括下载链接
> 权重，请参阅`follower_bots/ readme .md`的readme文件。

Dataset
-------

We are releasing a dataset of 560 games collected on Amazon mechanical turk. These are in 3 sections:

```
185 human-human games used to train the demonstration model
187 human-human games collected deploying the demo model on AWS mech turk.
188 human-model games collected deploying the demo model on AWS mech turk.
```

The dataset is [available for download here][3]. For data format documentation,
see our well-documentated schema definition at server/schemas/event.py. JSON files
are serialized from the Sqlite database, and contain the same schema.



Resources
---------

`resources.txt`: Links to resources that were useful in development of this game.

`guidelines.txt`: Guiding thoughts on style, code review, and source code management. Always up for generous interpretation and change.


[0]: https://github.com/lil-lab/cerealbar
[1]: https://git-lfs.github.com
[2]: https://cb2.ai/
[3]: https://lil.nlp.cornell.edu/resources/cb2-base/cb2-base-data.tar.bz2

