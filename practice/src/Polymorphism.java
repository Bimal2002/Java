import java.util.Scanner;
class Animal1 {
    void sound1(){
        System.out.println("Animal makes sound");
    }
}
class Dog1 extends Animal1{
    @Override   // it will override the parents method
    void sound1(){
        System.out.println("Dog barks");
    }
}
public class Polymorphism {
    public static void main(String[] args){
        Animal1 obj = new Dog1();
        obj.sound1();

    }
}
