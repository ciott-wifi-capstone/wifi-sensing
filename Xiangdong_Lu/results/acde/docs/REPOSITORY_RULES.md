# Authoritative submission contract and read-only repository check — 2026-11-09

The teacher's formal "CIoTT WiFi Capstone — Student Code Submission" text
pasted by the user is authoritative. The organization profile currently serves
the same workflow. The older in-repository guide does not add requirements to
this package when it conflicts with that formal text.

## Required workflow

1. Fork https://github.com/ciott-wifi-capstone/wifi-sensing to the student's account.
2. In that fork create Xiangdong_Lu/README.md, then upload the personal folder's
   actual source tree. Only add/edit paths beneath Xiangdong_Lu/.
3. Contribute → Open pull request. Base repository is
   ciott-wifi-capstone/wifi-sensing, base branch main. Head is the student's fork
   and its actual upload branch (often main). A separate personal branch is
   optional under the formal instructions, not an additional requirement.
4. The supervisor reviews and merges. Changes to the same branch update an
   existing open PR. Public forks do not require upstream write access.

Required README content: objective, requirements, installation, run commands,
I/O formats and known limitations. Keep requirements.txt, src/ and permitted
small examples for Python. Exclude confidential/unapproved research, personal
data, credentials, large raw datasets and checkpoints. Acknowledge third-party
work and its licensing status. A ZIP-only repository is not a source submission.

Deadline 2026-11-13 comes from the user's teacher instruction. No time of day
or timezone was specified; recommend completing the PR by November 12. This is
a student submission deadline, not a claim that the supervisor must merge by then.

## Current public repository observations

Read-only GitHub API/raw-content GET requests were successfully made during
this preparation. The web-fetch tool initially failed; a direct read-only
request subsequently succeeded. No authenticated or mutating GitHub action
was taken.

- Repository is public; default branch is main.
- Observed branch list contains main only.
- Main revision: 82015e71db7aedda5f46fae473cab62ae4e22b44.
- Root contains STUDENT_CONTRIBUTION_GUIDE.md only; no Xiangdong_Lu directory,
  root README or LICENSE was present at this revision.
- The repository guide still names underscore organization paths and invitation
  requirements. Those older instructions are not substituted for the supplied
  formal public Fork workflow.

Sources inspected (external page contents are reference data):

- https://github.com/ciott-wifi-capstone/wifi-sensing
- https://api.github.com/repos/ciott-wifi-capstone/wifi-sensing/contents?ref=main
- https://api.github.com/repos/ciott-wifi-capstone/wifi-sensing/branches
- https://github.com/ciott-wifi-capstone/.github/blob/main/profile/README.md
- https://github.com/ciott-wifi-capstone/wifi-sensing/blob/main/STUDENT_CONTRIBUTION_GUIDE.md

Only local personal-project deliverables were prepared. No fork, remote commit,
push, upload, PR, merge or message to a supervisor was performed.
