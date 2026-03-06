import java.util.Scanner;
//without Encapsulation(Bad)
class Student1{
    String name;
    int age;
}
//s.age = -100 // wrong data this way one can assign wrong data


// with proper Encapsulation
class Student2{
    private String name;
    private int age;
    public void setAge(int age){
        if(age>0){
            this.age = age;
        }
    }
    public int getAge(){
        return age;
    }
}// data is protected

public class Encapsulation {
    public static void main(String[] args){
        Student2 s = new Student2();
        s.setAge(20);
        int x= s.getAge();
        System.out.println(x);
    }
}
