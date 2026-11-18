# Xiangdong Lu：GitHub sensing 提交指南

准备日期：2026-11-09。**截止日期：2026-11-13**，老师未给出具体时刻或时区，建议最迟 11 月 12 日完成上传和 PR，留出修改时间。本包只做本地准备；以下 Fork、commit、push、创建 PR 的动作都由你自行操作。

## 1. 认准提交位置和规则

- 原仓库：[ciott-wifi-capstone/wifi-sensing](https://github.com/ciott-wifi-capstone/wifi-sensing)。类别为 **sensing**。
- 唯一个人目录：`Xiangdong_Lu/`，位于仓库根目录。
- 严格按照你贴出的老师正式规范：**Fork → 创建个人 README → 上传代码 → Contribute → Open pull request → supervisor 审核合并**。
- 无需原仓库写权限或邀请；需要自己的 GitHub 账号。
- 仅增改 `Xiangdong_Lu/` 内的文件，不改共享 README、指南或他人目录。
- 正式规范允许 head 使用本人 Fork 中含有修改的分支（通常为 `main`）；**没有强制另建个人分支**。

2026-11-09 只读核验时，原仓库为公开仓库、默认分支 `main`，根目录仅有 `STUDENT_CONTRIBUTION_GUIDE.md`，尚无 `Xiangdong_Lu/`。该旧指南含旧组织地址与邀请流程；本包以老师刚贴出的正式规范为准，详见 [规则记录](docs/REPOSITORY_RULES.md)。

## 2. 解压并认识最终结构

解压 `Xiangdong_Lu_WiCAL_Sensing_GitHub_Submission.zip`，得到一个顶层文件夹 `Xiangdong_Lu`。上传真实源码文件树，**不能只上传 ZIP**。最终仓库应显示：

```text
wifi-sensing/
├── STUDENT_CONTRIBUTION_GUIDE.md  ← 上游原文件，不修改
└── Xiangdong_Lu/
    ├── README.md
    ├── requirements.txt
    ├── src/
    ├── tests/
    ├── examples/
    ├── config/
    ├── results/
    ├── docs/
    ├── .gitignore
    └── ...
```

![网页提交位置与 base/head 核对示意图；非实际网页截图](docs/submission_workflow_cn.png)

## 3. 网页提交：逐步照做

1. 登录自己的 GitHub，打开上面的 **wifi-sensing** 原仓库。
2. 点击右上角 **Fork**，Owner 选择自己的账号，再点 **Create fork**。
3. 确认地址为 `github.com/你的账号/wifi-sensing`。后续修改都在本人 Fork；可直接使用 Fork 的 `main` 分支。
4. 在本人 Fork 根目录选择 **Add file → Create new file**。文件名准确填 `Xiangdong_Lu/README.md`。
5. 用文本编辑器打开本包 `Xiangdong_Lu/README.md`，复制**完整内容**到 GitHub 编辑区；点 **Commit changes**，提交信息可填 `Add Xiangdong Lu project README`。这一步创建个人目录。
6. 进入 Fork 中的 **`Xiangdong_Lu/`**，选择 **Add file → Upload files**。
7. 从本机解压后的 `Xiangdong_Lu` **里面**选择其文件与子文件夹（例如 `src`、`tests`、`examples`、`config`、`results`、`docs` 等），拖进上传区。README 已在第 5 步按本包完整内容建立，无需重复上传。**此时不要再拖最外层 `Xiangdong_Lu` 文件夹**，以免形成 `Xiangdong_Lu/Xiangdong_Lu/`。
8. 检查路径：应是 `Xiangdong_Lu/src/retest.py`，不能是仓库根目录 `src/retest.py`。检查 `.gitignore` 和 `.gitattributes`；若文件选择器隐藏点文件，可在个人目录用 **Create new file** 按原名建立并复制本包内容。若子目录未被保留，使用下面 Git 方法。
9. 填有意义的提交说明，例如 `Add WiCAL source, tests and reproducibility files`，再点 **Commit changes**。浏览器有批次限制时分批上传，保持目录结构，全部传完再开 PR。
10. 在本人 Fork 点击 **Contribute → Open pull request**（也可能显示 **Compare & pull request**）。需要时选 **compare across forks**。
11. 核对以下四项：

| GitHub 字段 | 正确值 |
|---|---|
| base repository | `ciott-wifi-capstone/wifi-sensing` |
| base branch | `main` |
| head repository | `你的账号/wifi-sensing` |
| compare branch | 本人 Fork 中实际上传代码的分支；按上述步骤为 `main` |

12. 点 **Files changed**：每个增改路径必须以 `Xiangdong_Lu/` 开头；不得有 ZIP、原始 CSI、论文或共享文件修改。
13. PR 标题填写 **`Add Xiangdong Lu's WiFi sensing capstone project`**；正文可复制 [PR_DESCRIPTION.md](PR_DESCRIPTION.md)。
14. 由你点 **Create pull request**，保存 PR 链接，按课程要求交给导师。Fork 中有文件并不等于完成向原仓库提交，必须有 PR。不要自行合并，等 supervisor 审核。
15. 如需修改，继续更新本人 Fork **同一分支**中的个人文件；原 PR 会自动更新，不用为每次修复另开 PR。

如果你此前已建立本任务的 Fork、分支或未合并 PR，沿用已有分支和 PR；compare branch 应填该分支的实际名称，不重复创建。

## 4. 网页无法保留子目录时：本地 Git 备选

以下命令仅供**你自行执行**。先按第 3 节完成 Fork 和 `Xiangdong_Lu/README.md` 的创建，再克隆**本人 Fork**。将 `YOUR_GITHUB_USERNAME` 替换成自己的账号：

```text
git clone https://github.com/YOUR_GITHUB_USERNAME/wifi-sensing.git
cd wifi-sensing
```

默认在本人 Fork 的 `main`；若已有本任务分支，先 `git switch 分支名`。把解压后的完整 `Xiangdong_Lu` 文件夹复制到克隆仓库根目录，同名 README 使用本包内容。不要复制 ZIP 或覆盖共享文件。原仓库如后来出现同名个人目录，先比较内容再更新，保留不属于本次准备的改动。

```text
git status --short
git add -- Xiangdong_Lu/
git diff --cached --name-only
git diff --cached --stat
git commit -m "Add Xiangdong Lu WiCAL sensing source and tests"
git push origin main
```

仅在实际使用本人 Fork 的 `main` 时执行上面的 push；已有个人分支时，把最后一行的 `main` 换成实际分支名。确认 `origin` 是本人 Fork，暂存文件全部在 `Xiangdong_Lu/` 内，再提交。然后回到网页按第 10–15 步创建 PR。

Git 身份使用自己已确认的配置；不要把邮箱、令牌或密码写进项目。缺少写权限时应检查是否误操作原仓库，不需要申请原仓库写权限才能发公开 Fork 的 PR。

## 5. 本包的运行与公开范围

- 运行说明见英文 [README.md](README.md)，实测核验见 [VALIDATION.md](VALIDATION.md)。
- 包含真实项目源码、配置、经去标识的数值结果和明确标注的合成示例；不带真实原始 CSI、私有采集日志、论文或私人身份信息。
- 原始 Widar2 Python 文件的公开再分发许可未核实，未打包；归属和许可状态见 [NOTICE.md](NOTICE.md) 与 [LICENSE](LICENSE)。可选原核心核对需单独取得可合法使用的文件。
- 19.70°／7.28° 和约 63% 下降仅为所选四点、暂定映射与角度约定下的条件结果。不能改写为已经验证的绝对人体定位精度。
- `E` 对应 `B_new`，不是第五点；旧 B 已被替换，每点只有一次静态有人 Run。物理天线方向、AntSEL 与人体路径身份仍未确认。

**在 2026-11-13 前由你完成 PR 提交；supervisor 之后审核合并。**
