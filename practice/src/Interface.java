import java.util.Scanner;
interface Animal3{
    void sound();
}
class Dog3 implements Animal3{
    public void sound(){
        System.out.println("Dog barks");
    }
}
interface A{
    void methodA();
}
interface B{
    void methodB();
}
class C implements A,B{
    public void methodA(){
        System.out.println("A");
    }
    public void methodB(){
        System.out.println("B");
    }
}
public class Interface {
    public static void main(String[] args){
        Dog3 d = new Dog3();
        d.sound();
        C c = new C();
        c.methodA();
        c.methodB();
    }
}
