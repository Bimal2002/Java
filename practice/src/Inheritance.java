import java.util.Scanner;
import java.util.ArrayList;
class Parent{
    //Properties
    String type = "Biswanath Gayali";
}
class Child extends Parent{
    //additional properties
    void printType(){
        System.out.println(super.type);
    }
}
class Animal{
    void eat(){
        System.out.println("Eating...");
    }
}
class Dog extends Animal{
    void bark(){
        System.out.println("Barking ...");
    }
}
public class Inheritance {
    public static void main(String[] args){
        Dog d = new Dog();
        d.eat();//inherited
        d.bark();

        Child c = new Child();
        c.printType();
    }
}
