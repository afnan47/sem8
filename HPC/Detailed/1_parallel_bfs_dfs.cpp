#include<bits/stdc++.h>
using namespace std;
 
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
 
void bfs(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();
        cout << node->val << " ";
        if (node->left) {
            q.push(node->left);
        }
        if (node->right) {
            q.push(node->right);
        }
    }
}

void dfs(TreeNode* root) {
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        TreeNode* node = s.top();
        s.pop();
        cout << node->val << " ";
        if (node->right) {
            s.push(node->right);
        }
        if (node->left) {
            s.push(node->left);
        }
    }
}
 
 
void parallel_bfs(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int qSize = q.size();
        #pragma omp parallel for
        for (int i = 0; i < qSize; i++) {
            TreeNode* node;
            #pragma omp critical
            {
                node = q.front();
                q.pop();
                cout << node->val << " ";
            }
            if (node->left)  q.push(node->left);        
            if (node->right) q.push(node->right);
        }
    }
}
 
void parallel_dfs(TreeNode* root) {
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        int sSize = s.size();
        #pragma omp parallel for
        for (int i = 0; i < sSize; i++) {
            TreeNode* node;
            #pragma omp critical
            {
                node = s.top();
                s.pop();
                cout << node->val << " ";
            }
            if (node->right) s.push(node->right);
            if (node->left) s.push(node->left);
        }
    }
}
 


 
int main() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);
 
    cout << "BFS traversal: ";
    auto start = chrono::high_resolution_clock::now();
    bfs(root);
    auto end = chrono::high_resolution_clock::now();
    cout << "\nBFS took " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds." << endl;
    cout << endl;
 
    cout << "Parallel BFS traversal: ";
    start = chrono::high_resolution_clock::now();
    parallel_bfs(root);
    end = chrono::high_resolution_clock::now();
    cout << "\nParallel BFS took " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds." << endl;
 
    cout << "---------------------------------------------------------"<<endl;
 
    cout << "DFS traversal: ";
    start = chrono::high_resolution_clock::now();
    dfs(root);
    end = chrono::high_resolution_clock::now();
    cout << "\nDFS took " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds." << endl;
    cout << endl;
 
    cout << "Parallel DFS traversal: ";
    start = chrono::high_resolution_clock::now();
    parallel_dfs(root);
    end = chrono::high_resolution_clock::now();
    cout << "\nParallel DFS took " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds." << endl;
 
 
    return 0;
}