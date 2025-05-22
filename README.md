# COSC81-Final-Project

---

SUBMODULE INSTRUCTIONS
I (Megan) forked the VNC-ROS repo by AQL and wanted to use the docker container/easy access to previous code for the final project. Mihir created a repo for our final project, which I made into a submodule: a repo inside a repo. In the parent repo, the child repo looks like an empty folder without setup. But after setting up, the folder looks like the remote branch. Any commits made within that folder update only that child repo, and any commits made outside that folder update only the parent repo.
Here's how to set up.

1. Within workspace/src, run

```
git submodule add https://github.com/mihirBSingh/COSC81-Final-Project.git cs81-finalproj
```

2. Check it worked by running

```
git submodule status
```

In Cursor/VSCode with the Git version control extension, a green "A" should appear next to the submodule, denoting a new submodule added to the index. After changes in the submodule, there should be a blue "S" denoting submodule having staged changes.
