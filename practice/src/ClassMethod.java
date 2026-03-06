import java.util.Scanner;

//class student{
//    String name;
//    int age;
//    student(String a,int b){
//        name = a;
//        age = b;
//    }
//    void display(){
//        System.out.println(name+ " " + age);
//    }
//}
class Student{
    static String college = "IIT";
    String name ;
    Student(String name){
        this.name = name;
    }
    void display(){
        System.out.println(name + " "+ college);
    }
}

// Bank Account
class BankAccount{
    String accHolder ;
    int accNo ;
    int balance;
    BankAccount(String name,int id,int initialBal){
        this.accHolder = name;
        this.accNo =id;
        this.balance = initialBal;
    }
    //deposit
    void deposit(int amt){
        if(amt>0){
            balance += amt;
            System.out.println("Deposited: "+ amt);
        }
    }
    // withdraw money
    void withdraw(int amt){
        if(amt <= balance){
            balance -= amt;
            System.out.println("Withdrawn : "+amt);
        }else{
            System.out.println("Insuffient Balance ");
        }
    }
    void display(){
        System.out.println("Holder: "+accHolder + "| A/C: "+accNo +" | Balance: " + balance);
    }
}

public class ClassMethod {
   public static void main(String[] args){
//       student s1 = new student();
//       s1.name = "Bimal";
//       s1.age = 23;
//       s1.display();
//       student s2 = new student("Bimal",23);
       Student s3 = new Student("Bimal");
       s3.display();

       // Using the new BankAccount logic
       BankAccount myAcc = new BankAccount("Bimal", 101, 5000);
       myAcc.deposit(1500);
       myAcc.withdraw(2000);
       myAcc.display();
   }
}
