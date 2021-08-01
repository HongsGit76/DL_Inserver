#include <iostream>

using namespace std;

struct student {
    int id;
    char *name;
    float percentage;

    void show(){
        cout << "아이디: " << id << endl;
        cout << "이름: " << name << endl;
        cout << "확률: " << percentage << endl;
    }
};



int main() {
    student s = {1,"철수", 10.5};

    s.show();
    return 0;
}