import java.util.Scanner;
abstract class Shape{
    abstract void area();
    void display(){
        System.out.println("Shape class");
    }
}
// child class must implement the abstract method
class Circle extends Shape{
    void area(){
        System.out.println("Area of circle");
    }
}
public class Abstraction {
    public static void main(String[] args){
        Shape s = new Circle();
        s.area();
    }
}
